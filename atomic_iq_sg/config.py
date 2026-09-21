from __future__ import annotations
import numpy as np

LOOKBACK = 20
COST_BPS = 10.0
COST_RATE = COST_BPS / 10000.0
CALIBRATION_END = '1999-12-31'
EVALUATION_START = '2000-01-01'
EVALUATION_END = '2024-12-31'
TARGET_MIN_EIG = 1e-6
BOOTSTRAP_REPS = 10000
BOOTSTRAP_BLOCK = 12
BOOTSTRAP_SEED = 20260920

# State-Gram design: chi is fixed structurally at the strongest interior
# coupling on the retained grid.  c, delta and p are calibrated pre-2000.
PRIMARY_CHI = 0.90
STATE_SCALE = 'mad'
C_GRID = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7],float)
DELTA_GRID = np.array([0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0],float)
P_GRID = np.array([0.0,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1.0],float)
GS_THRESHOLD_GRID = np.round(np.linspace(0.0,1.0,101),2)
SRE_ALPHA_GRID = np.round(np.linspace(0.0,1.0,101),2)
