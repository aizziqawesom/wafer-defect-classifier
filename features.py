"""
utils/features.py
Hand-crafted feature extraction for wafer map classification.
Mirrors the feature engineering in the main notebook.
"""

import numpy as np
from scipy import ndimage


def extract_features(wmap: np.ndarray) -> np.ndarray:
    """
    Extract 27-dimensional feature vector from a normalized 64x64 wafer map.

    Features:
        [0:9]   Zone density (3x3 grid)
        [9:14]  Radial ring density (5 rings, center→edge)
        [14:22] Angular sector density (8 sectors)
        [22]    Global defect density
        [23]    Density standard deviation
        [24]    Max pixel value (defect presence flag)
        [25]    Number of defect clusters (connected components)
        [26]    Padding (reserved)

    Args:
        wmap: Normalized wafer map array, shape (64, 64), values in [0, 1].
              Defective pixels are expected at > 0.75.

    Returns:
        Feature vector of shape (27,).
    """
    h, w = wmap.shape
    cx, cy = h // 2, w // 2
    defect_mask = (wmap > 0.75).astype(np.float32)
    features = []

    # Zone density (3x3 grid)
    zone_size = h // 3
    for r in range(3):
        for c in range(3):
            zone = defect_mask[r*zone_size:(r+1)*zone_size, c*zone_size:(c+1)*zone_size]
            features.append(float(zone.mean()))

    # Radial ring density (5 rings)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_r = min(cx, cy)
    for ring in range(5):
        r_min = ring * max_r / 5
        r_max = (ring + 1) * max_r / 5
        ring_mask = (dist >= r_min) & (dist < r_max)
        features.append(float(defect_mask[ring_mask].mean()) if ring_mask.sum() > 0 else 0.0)

    # Angular sector density (8 sectors)
    angles = np.arctan2(Y - cy, X - cx)
    for sector in range(8):
        a_min = -np.pi + sector * np.pi / 4
        a_max = -np.pi + (sector + 1) * np.pi / 4
        sector_mask = (angles >= a_min) & (angles < a_max)
        features.append(float(defect_mask[sector_mask].mean()) if sector_mask.sum() > 0 else 0.0)

    # Global statistics
    features.append(float(defect_mask.mean()))
    features.append(float(defect_mask.std()))
    features.append(float(defect_mask.max()))
    _, n_clusters = ndimage.label(defect_mask)
    features.append(float(n_clusters))

    # Pad to 27
    while len(features) < 27:
        features.append(0.0)

    return np.array(features[:27], dtype=np.float32)


def extract_batch(wmaps: np.ndarray) -> np.ndarray:
    """Extract features for a batch of wafer maps."""
    return np.stack([extract_features(w) for w in wmaps])
