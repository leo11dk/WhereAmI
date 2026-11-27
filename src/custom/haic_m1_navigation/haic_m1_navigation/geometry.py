"""Small planar geometry helpers. Angles are radians; transforms map child to parent."""
from __future__ import annotations
import math
import numpy as np


def wrap_angle(angle):
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def classify_scan_stamp(stamp, last_stamp, epoch_start, ros_now,
                        clock_tolerance_sec=.25, future_tolerance_sec=.5):
    """Classify time without confusing a delayed packet with a clock reset.

    Clock and sensor messages use different ROS subscriptions; a bounded small
    delivery skew is tolerated. A scan from a previous simulator clock must
    not initialize a new odometry epoch before the current /clock catches up.
    """
    values = [stamp, epoch_start, ros_now]
    if last_stamp is not None:
        values.append(last_stamp)
    if not all(math.isfinite(float(v)) for v in values):
        return 'invalid'
    if stamp > ros_now + future_tolerance_sec:
        return 'wait_clock'
    if (ros_now < epoch_start-clock_tolerance_sec
            or (last_stamp is not None and ros_now < last_stamp-clock_tolerance_sec)):
        return 'clock_reset'
    if stamp < epoch_start-1e-9:
        return 'before_epoch'
    if last_stamp is not None and stamp <= last_stamp:
        return 'duplicate' if stamp == last_stamp else 'out_of_order'
    return 'accept'


def se2_matrix(pose):
    x, y, yaw = np.asarray(pose, dtype=float)
    if not np.all(np.isfinite([x, y, yaw])):
        raise ValueError('pose must contain three finite values')
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0., 0., 1.]])


def matrix_pose(transform):
    t = np.asarray(transform, dtype=float)
    if t.shape != (3, 3) or not np.all(np.isfinite(t)):
        raise ValueError('expected a finite SE2 matrix')
    return np.array([t[0, 2], t[1, 2], math.atan2(t[1, 0], t[0, 0])])


def transform_points(transform, points):
    t, p = np.asarray(transform, dtype=float), np.asarray(points, dtype=float)
    if t.shape != (3, 3) or p.ndim != 2 or p.shape[1] != 2:
        raise ValueError('expected a 3x3 transform and Nx2 points')
    return p @ t[:2, :2].T + t[:2, 2]


def scan_points(scan, max_points=720):
    """Convert a LaserScan-like message, preserving finite measured bearings."""
    ranges = np.asarray(scan.ranges, dtype=float)
    angle_min, increment = float(scan.angle_min), float(scan.angle_increment)
    if not np.isfinite([angle_min, increment]).all() or increment == 0:
        raise ValueError('invalid scan angles')
    if max_points < 3:
        raise ValueError('max_points must be >=3')
    valid = np.flatnonzero(np.isfinite(ranges) & (ranges >= scan.range_min)
                          & (ranges <= scan.range_max))
    if len(valid) > max_points:
        valid = valid[np.linspace(0, len(valid)-1, max_points).astype(int)]
    angles = angle_min + increment * valid
    return np.column_stack((ranges[valid]*np.cos(angles), ranges[valid]*np.sin(angles)))


def pelvis_from_scan_pose(scan_pose, pelvis_from_scan):
    """Recover odom<-pelvis from planar odom<-level_scan and full extrinsic.

    pelvis_from_scan is the timestamped TF pelvis<-level_scan. Its rotation
    contains gravity tilt and any waist motion. The pelvis Z gauge is zero;
    only XY is observable from this planar estimator. No second scan parent TF
    is published. Return full 4x4 odom<-pelvis.
    """
    pose = np.asarray(scan_pose, dtype=float)
    extrinsic = np.asarray(pelvis_from_scan, dtype=float)
    if pose.shape != (3,) or extrinsic.shape != (4, 4):
        raise ValueError('expected pose[3] and pelvis<-scan 4x4 transform')
    if not np.isfinite(pose).all() or not np.isfinite(extrinsic).all():
        raise ValueError('nonfinite transform')
    c, s = math.cos(pose[2]), math.sin(pose[2])
    r_scan = np.array([[c,-s,0], [s,c,0], [0,0,1.]])
    r_pelvis = r_scan @ extrinsic[:3,:3].T
    out = np.eye(4)
    out[:3,:3] = r_pelvis
    out[:2,3] = pose[:2] - (r_pelvis @ extrinsic[:3,3])[:2]
    return out
