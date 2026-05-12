"""Shared constants for the CNN oscillation-analysis pipeline."""

import numpy as np

N_FILAMENT_WORKERS = 64
N_PIXEL_WORKERS = 64
CNN_FREQUENCY_GRID = np.linspace(1 / (1440 * 60), 1 / 120, 2000).astype(np.float32)
CP_DELTA = 0.0000006
DETECTION_FMIN_HZ = 1.0 / (3.0 * 3600.0)
DETECTION_FMAX_HZ = 1.0e-3
