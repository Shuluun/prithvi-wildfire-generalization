"""M5 decoder comparison — model definitions and shared training plumbing.

M5 distinguishes three hypotheses for the weak M4 linear-probe result:
  A. nonlinear readout failure      -> pointwise MLP (M5a)
  B. need for spatial decoding      -> lightweight spatial decoder (M5b)
  C. frozen representation lacks     -> spectral spatial CNN control + ranking
     transferable burn semantics

All models consume the frozen M4a feature cache (or the raw spectral stack for
the matched control) and use the IDENTICAL event-disjoint K5 protocol as M2/M4.
"""
