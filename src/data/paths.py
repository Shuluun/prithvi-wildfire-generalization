"""Project paths, driven by environment variables.

Raw data (large, gitignored) lives OUTSIDE this repo, by default in a sibling
``data/`` directory next to the repo's parent. Set ``PRITHVI_WF_ROOT`` to point
at that storage root to override. Small metadata CSVs and results live INSIDE
the repo and are resolved relative to the repo root, so the checkout is
portable across machines and directory names.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Raw data root: large HLS/MTBS rasters, NOT in git.
PROJECT_ROOT = os.environ.get("PRITHVI_WF_ROOT", os.path.dirname(REPO_ROOT))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
RAW_ROOT = os.path.join(DATA_ROOT, "raw")

METADATA_ROOT = os.path.join(REPO_ROOT, "data", "metadata")  # small CSVs (in git)
RESULTS_ROOT = os.path.join(REPO_ROOT, "results")

HLS_BURN_SCARS_DIR = os.path.join(RAW_ROOT, "hls_burn_scars")
MTBS_DIR = os.path.join(RAW_ROOT, "mtbs")
MTBS_ZIP = os.path.join(MTBS_DIR, "mtbs_perims_DD.zip")
MTBS_SHP_DIR = os.path.join(MTBS_DIR, "mtbs_perims_DD")
