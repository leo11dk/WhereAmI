"""ROS-independent geometry, tilt estimation and PointCloud2 decoding.

The tilt filter estimates the *up direction*, never global yaw or XY position.
Accelerometer measurements are specific force: when stationary they point up.
Walking acceleration can bias tilt, so outputs must be verified on the robot.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


def unit(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.all(np.isfinite(value)) or norm < 1e-10:
        raise ValueError('expected a nonzero finite 3-vector')
    return value / norm


def rotation_vector_matrix(vector: Sequence[float]) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError('expected a finite rotation vector')
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = v / angle
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


def quaternion_matrix(xyzw: Sequence[float]) -> np.ndarray:
    q = np.asarray(xyzw, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-12:
        raise ValueError('expected a nonzero finite quaternion')
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ])


def matrix_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Return an xyzw quaternion; handle rotations near 180 degrees too."""
    r = np.asarray(matrix, dtype=float)
    if r.shape != (3, 3) or not np.all(np.isfinite(r)):
        raise ValueError('expected a finite 3x3 rotation')
    # Symmetric eigenproblem, avoiding division by a small quaternion component.
    k = np.array([
        [r[0,0]-r[1,1]-r[2,2], r[1,0]+r[0,1], r[2,0]+r[0,2], r[2,1]-r[1,2]],
        [r[1,0]+r[0,1], r[1,1]-r[0,0]-r[2,2], r[2,1]+r[1,2], r[0,2]-r[2,0]],
        [r[2,0]+r[0,2], r[2,1]+r[1,2], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
        [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], r.trace()],
    ]) / 3.0
    _, eigvec = np.linalg.eigh(k)
    q = eigvec[:, -1]
    return -q if q[3] < 0 else q


def level_rotation(up_in_sensor: Sequence[float]) -> np.ndarray:
    """R_level_sensor: Z up, X the sensor's forward axis projected horizontally.

    Origins coincide. This removes tilt while preserving local heading; the
    output frame still turns/translates with the robot and is NOT map or odom.
    """
    z = unit(up_in_sensor)
    x = np.array([1., 0., 0.]) - z[0] * z
    if np.linalg.norm(x) < 1e-5:
        raise ValueError('sensor forward axis is vertical; robot may have fallen')
    x = unit(x)
    y = np.cross(z, x)
    return np.stack((x, y, z))


def read_xyz(cloud) -> np.ndarray:
    """Decode XYZ, honoring field offsets, stride, row padding and byte order.

    Accepts a sensor_msgs/PointCloud2 or a duck-typed fixture. Does not assume
    every record is 12 bytes: HAIC records also contain the uint16 ring field.
    """
    width, height = int(cloud.width), int(cloud.height)
    if width < 0 or height < 0:
        raise ValueError('negative cloud dimensions')
    if width == 0 or height == 0:
        return np.empty((0, 3), dtype=float)
    point_step, row_step = int(cloud.point_step), int(cloud.row_step)
    fields = {field.name: field for field in cloud.fields}
    formats, offsets = [], []
    order = '>' if cloud.is_bigendian else '<'
    # sensor_msgs/PointField.FLOAT32=7, FLOAT64=8.
    for name in ('x', 'y', 'z'):
        field = fields.get(name)
        if field is None or field.count != 1 or field.datatype not in (7, 8):
            raise ValueError('cloud requires scalar float32/float64 x,y,z fields')
        size = 4 if field.datatype == 7 else 8
        if field.offset < 0 or field.offset + size > point_step:
            raise ValueError('XYZ field extends beyond point_step')
        formats.append(order + ('f4' if size == 4 else 'f8'))
        offsets.append(int(field.offset))
    if point_step <= 0 or row_step < point_step * width:
        raise ValueError('invalid cloud strides')
    data = memoryview(cloud.data)
    if data.nbytes < row_step * height:
        raise ValueError('cloud data is shorter than row_step * height')
    dtype = np.dtype(dict(names=['x','y','z'], formats=formats,
                         offsets=offsets, itemsize=point_step))
    rows = np.ndarray((height, width), dtype=dtype, buffer=data,
                      strides=(row_step, point_step))
    return np.column_stack([rows[name].reshape(-1) for name in ('x','y','z')]).astype(float)


@dataclass(frozen=True)
class ScanConfig:
    bins: int = 720
    range_min: float = 0.30
    range_max: float = 30.0
    min_height: float = -0.15
    max_height: float = 0.15

    def __post_init__(self):
        if self.bins < 8 or self.bins > 10000:
            raise ValueError('scan bins must be in [8,10000]')
        if not all(math.isfinite(v) for v in (self.range_min,self.range_max,
                                              self.min_height,self.max_height)):
            raise ValueError('scan limits must be finite')
        if not 0 <= self.range_min < self.range_max:
            raise ValueError('invalid scan range limits')
        if self.min_height >= self.max_height:
            raise ValueError('invalid scan height limits')


def make_scan(points_level: np.ndarray, config: ScanConfig):
    """Minimum horizontal range per bearing, using a height band at LiDAR level.

    Empty bins remain +inf (unknown/no return), not zero or invented walls.
    Distances are from the REAL LiDAR origin, never from the pelvis.
    """
    p = np.asarray(points_level, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError('expected Nx3 points')
    finite = np.all(np.isfinite(p), axis=1)
    p = p[finite]
    distance = np.hypot(p[:,0], p[:,1])
    valid = ((p[:,2] >= config.min_height) & (p[:,2] <= config.max_height)
             & (distance >= config.range_min) & (distance <= config.range_max))
    p, distance = p[valid], distance[valid]
    increment = 2 * math.pi / config.bins
    index = np.rint((np.arctan2(p[:,1], p[:,0]) + math.pi) / increment).astype(int)
    index %= config.bins
    ranges = np.full(config.bins, np.inf, dtype=np.float32)
    np.minimum.at(ranges, index, distance)
    return ranges, int(valid.sum())


class TiltEstimator:
    """Gyro-propagated up direction with gated accelerometer correction.

    Initializes while nearly stationary, using IMU time (not wall time).
    There is no velocity integration, XY integration or absolute yaw output.
    """

    def __init__(self, calibration_sec=1.0, tau_sec=2.0, gravity=9.81,
                 accel_tolerance=1.5, calibration_gyro_limit=0.30,
                 calibration_accel_std_limit=0.8, max_gap_sec=0.20):
        vals = (calibration_sec,tau_sec,gravity,accel_tolerance,
                calibration_gyro_limit,calibration_accel_std_limit,max_gap_sec)
        if not all(math.isfinite(v) and v > 0 for v in vals):
            raise ValueError('tilt filter parameters must be finite and positive')
        (self.calibration_sec,self.tau_sec,self.gravity,self.accel_tolerance,
         self.gyro_limit,self.std_limit,self.max_gap_sec) = vals
        self.reset()

    def reset(self):
        self.last_stamp = None
        self.start_stamp = None
        self.samples = []
        self.up = None
        self.state = 'CALIBRATING_HOLD_STILL'
        self.accel_correction_used = False

    def update(self, stamp, gyro, accel):
        gyro, accel = np.asarray(gyro,dtype=float), np.asarray(accel,dtype=float)
        if (not math.isfinite(stamp) or gyro.shape != (3,) or accel.shape != (3,)
                or not np.all(np.isfinite(gyro)) or not np.all(np.isfinite(accel))):
            raise ValueError('invalid IMU sample')
        if self.last_stamp is not None:
            dt = stamp - self.last_stamp
            if dt == 0:
                return None  # duplicate: do not advance filter or history
            if dt < 0 or dt > self.max_gap_sec:
                self.reset()
        dt = 0.0 if self.last_stamp is None else stamp - self.last_stamp
        self.last_stamp = stamp
        norm = float(np.linalg.norm(accel))
        accel_ok = abs(norm - self.gravity) <= self.accel_tolerance
        self.accel_correction_used = False
        if self.up is None:
            if not accel_ok or np.linalg.norm(gyro) > self.gyro_limit:
                self.start_stamp = None
                self.samples = []
                self.state = 'CALIBRATING_HOLD_STILL'
                return None
            if self.start_stamp is None:
                self.start_stamp = stamp
            self.samples.append(accel.copy())
            if stamp - self.start_stamp < self.calibration_sec or len(self.samples) < 5:
                return None
            if np.max(np.std(self.samples, axis=0)) > self.std_limit:
                self.start_stamp = stamp
                self.samples = [accel.copy()]
                return None
            self.up = unit(np.mean(self.samples, axis=0))
            self.samples = []
        else:
            # A fixed world vector expressed in a rotating sensor obeys -w×up.
            predicted = rotation_vector_matrix(-gyro * dt) @ self.up
            alpha = 1 - math.exp(-dt / self.tau_sec) if accel_ok else 0.0
            measured = accel / norm if accel_ok else predicted
            self.up = unit((1-alpha)*predicted + alpha*measured)
            self.accel_correction_used = bool(accel_ok)
        self.state = 'TILT_READY'
        return self.up.copy()
