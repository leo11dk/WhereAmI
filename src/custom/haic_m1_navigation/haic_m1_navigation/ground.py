"""Estimate the local floor from the leveled LiDAR cloud, never simulator Z."""
import numpy as np


def floor_height(points):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError('expected Nx3 leveled points')
    distance = np.hypot(p[:, 0], p[:, 1])
    z = p[(np.isfinite(p).all(axis=1) & (distance > 1.) & (distance < 7.)
           & (p[:, 2] > -2.) & (p[:, 2] < -.6)), 2]
    if len(z) < 30:
        raise ValueError('WAIT_OBSERVED_FLOOR')
    counts, edges = np.histogram(z, bins=np.arange(-2., -.57, .03))
    index = int(np.argmax(counts))
    support = z[(z >= edges[index]-.03) & (z <= edges[index+1]+.03)]
    if len(support) < 30:
        raise ValueError('WAIT_STABLE_FLOOR')
    return float(np.median(support))
