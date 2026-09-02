"""M5 model definitions: pointwise MLP, lightweight spatial decoder, spectral CNN.

Each model maps its input to a single burned-class logit map at the native
512x512 grid (the caller interpolates/decodes as needed). The frozen Prithvi
encoders are NOT trained here — only the probe/decoder heads have trainable
parameters. Every model exposes ``n_trainable`` and ``param_table()`` so the
exact architecture and parameter count are recorded before training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_params(layer):
    w = layer.weight.numel()
    b = 0 if layer.bias is None else layer.bias.numel()
    return w + b


class PointwiseMLP(nn.Module):
    """M5a — nonlinear POINTWISE readout on concatenated frozen layers [5,11,17,23].

    4096 -> Linear(256) -> GELU -> Dropout(0.1) -> Linear(64) -> GELU -> Linear(1)
    implemented as 1x1 convolutions (mathematically identical to a per-token MLP).
    No 3x3 conv, no spatial attention, no learned upsampling, no skip connections
    between tokens. Token logits (32x32) -> bilinear -> native 512x512 grid.
    """

    def __init__(self, in_ch=4096, hidden=256, mid=64, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Conv2d(in_ch, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, mid, kernel_size=1)
        self.fc3 = nn.Conv2d(mid, 1, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for m in (self.fc1, self.fc2, self.fc3):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)

    def forward(self, feat32):
        # feat32: (B, 4096, 32, 32)
        x = F.gelu(self.fc1(feat32))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return F.interpolate(x, size=(512, 512), mode="bilinear",
                             align_corners=False)

    @property
    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_table(self):
        return {
            "fc1 (4096->256, 1x1)": _conv_params(self.fc1),
            "fc2 (256->64, 1x1)": _conv_params(self.fc2),
            "fc3 (64->1, 1x1)": _conv_params(self.fc3),
            "dropout": 0,
        }


class SpatialDecoder(nn.Module):
    """M5b — deliberately lightweight spatial decoder on frozen layers [5,11,17,23].

    Per layer: 1x1 projection 1024 -> 64. Concatenate -> 256 channels. Then
    Conv3x3 256->128 (GELU), Conv3x3 128->64 (GELU), followed by progressive
    bilinear 2x upsampling with small 3x3 convolutional refinement, ending in a
    1-channel logit map at 512x512. No transformer/attention/UNet/UperNet/CRF/
    morphology/TTA/ensemble; no encoder fine-tuning. Total params ~0.66M (<= 1-2M).
    """

    def __init__(self, in_ch=1024, n_layers=4, proj_ch=64):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(in_ch, proj_ch, kernel_size=1) for _ in range(n_layers)])
        self.conv1 = nn.Conv2d(n_layers * proj_ch, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ref1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ref2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.ref3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.head = nn.Conv2d(8, 1, kernel_size=3, padding=1)

    def forward(self, feat):
        # feat: (B, 4, 1024, 32, 32)
        proj = [self.projections[i](feat[:, i]) for i in range(len(self.projections))]
        x = torch.cat(proj, dim=1)            # (B, 256, 32, 32)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))             # (B, 64, 32, 32)
        x = F.gelu(self.ref1(self._up(x)))    # 64
        x = F.gelu(self.ref2(self._up(x)))    # 128
        x = F.gelu(self.ref3(self._up(x)))    # 256
        return self.head(self._up(x))         # 512 -> (B, 1, 512, 512)

    @staticmethod
    def _up(x):
        return F.interpolate(x, scale_factor=2, mode="bilinear",
                             align_corners=False)

    @property
    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_table(self):
        out = {}
        for i, p in enumerate(self.projections):
            out[f"proj layer{i} (1024->64, 1x1)"] = _conv_params(p)
        out["conv1 (256->128, 3x3)"] = _conv_params(self.conv1)
        out["conv2 (128->64, 3x3)"] = _conv_params(self.conv2)
        out["ref1 (64->32, 3x3)"] = _conv_params(self.ref1)
        out["ref2 (32->16, 3x3)"] = _conv_params(self.ref2)
        out["ref3 (16->8, 3x3)"] = _conv_params(self.ref3)
        out["head (8->1, 3x3)"] = _conv_params(self.head)
        return out


class SpectralCNN(nn.Module):
    """Matched-capacity SPECTRAL spatial control (8 spectral features, native 512).

    Small symmetric convolutional segmentation model with local 3x3 context and
    ~0.52M parameters (comparable order to SpatialDecoder's ~0.66M). No skip
    connections, no extra engineered features, no dNBR, no augmentation. This is
    NOT an optimized spectral model — it is a matched-capacity control asking
    whether Prithvi adds value beyond what a lightweight spatial model extracts
    directly from the 6 bands + NDVI + NBR.
    """

    def __init__(self, in_ch=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, padding=1)          # 512
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)               # 256
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)              # 128
        self.down3 = nn.Conv2d(128, 128, 3, stride=2, padding=1)             # 64
        self.down4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)             # 32
        self.up1 = nn.Conv2d(128, 64, 3, padding=1)                          # 64
        self.up2 = nn.Conv2d(64, 64, 3, padding=1)                           # 128
        self.up3 = nn.Conv2d(64, 32, 3, padding=1)                           # 256
        self.up4 = nn.Conv2d(32, 16, 3, padding=1)                           # 512
        self.head = nn.Conv2d(16, 1, 3, padding=1)                           # 512

    def forward(self, x):
        # x: (B, 8, 512, 512)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.down1(x))
        x = F.gelu(self.down2(x))
        x = F.gelu(self.down3(x))
        x = F.gelu(self.down4(x))             # (B, 128, 32, 32)
        x = F.gelu(self.up1(self._up(x)))     # 64
        x = F.gelu(self.up2(self._up(x)))     # 128
        x = F.gelu(self.up3(self._up(x)))     # 256
        x = F.gelu(self.up4(self._up(x)))     # 512
        return self.head(x)                   # (B, 1, 512, 512)

    @staticmethod
    def _up(x):
        return F.interpolate(x, scale_factor=2, mode="bilinear",
                             align_corners=False)

    @property
    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_table(self):
        return {
            "conv1 (8->32, 3x3)": _conv_params(self.conv1),
            "down1 (32->64, s2)": _conv_params(self.down1),
            "down2 (64->128, s2)": _conv_params(self.down2),
            "down3 (128->128, s2)": _conv_params(self.down3),
            "down4 (128->128, s2)": _conv_params(self.down4),
            "up1 (128->64, 3x3)": _conv_params(self.up1),
            "up2 (64->64, 3x3)": _conv_params(self.up2),
            "up3 (64->32, 3x3)": _conv_params(self.up3),
            "up4 (32->16, 3x3)": _conv_params(self.up4),
            "head (16->1, 3x3)": _conv_params(self.head),
        }
