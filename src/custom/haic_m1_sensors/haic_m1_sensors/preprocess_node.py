"""ROS 2 adapter: raw cloud + colocated IMU -> level cloud and wall scan."""

from __future__ import annotations

from array import array
from collections import deque
import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu, LaserScan, PointCloud2, PointField
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from .core import (
    ScanConfig, TiltEstimator, level_rotation, make_scan, matrix_quaternion,
    quaternion_matrix, read_xyz,
)


def stamp_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def frame_name(name):
    return name.lstrip('/')


class SensorPreprocess(Node):
    def __init__(self):
        super().__init__('haic_sensor_preprocess')
        defaults = {
            'cloud_topic': '/lidar/points', 'imu_topic': '/lidar/imu',
            'cloud_frame': 'lidar_link', 'imu_frame': 'imu_link',
            'level_frame': 'haic_lidar_level', 'scan_topic': '/scan',
            'level_cloud_topic': '/haic/lidar/points_level', 'scan_bins': 720,
            'range_min_m': 0.3, 'range_max_m': 30.0,
            'min_height_m': -0.15, 'max_height_m': 0.15,
            'calibration_sec': 1.0, 'tilt_tau_sec': 2.0, 'gravity_mps2': 9.81,
            'accel_tolerance_mps2': 1.5, 'calibration_gyro_limit_rps': 0.3,
            'calibration_accel_std_limit': 0.8, 'max_imu_gap_sec': 0.2,
            'max_imu_cloud_skew_sec': 0.08, 'max_wall_input_age_sec': 3.0,
            'max_tilt_deg': 50.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in defaults}
        self.cloud_frame = frame_name(self.p['cloud_frame'])
        self.imu_frame = frame_name(self.p['imu_frame'])
        self.level_frame = frame_name(self.p['level_frame'])
        if (not self.level_frame or self.level_frame in
                (self.cloud_frame,self.imu_frame,'map','odom','world','pelvis','base_link','base_footprint')):
            raise ValueError('level_frame must be a distinct local sensor frame, not a navigation frame')
        for name in ('max_imu_cloud_skew_sec', 'max_wall_input_age_sec', 'max_tilt_deg'):
            if not math.isfinite(self.p[name]) or self.p[name] <= 0:
                raise ValueError(name + ' must be finite and positive')
        self.scan_config = ScanConfig(
            bins=self.p['scan_bins'], range_min=self.p['range_min_m'],
            range_max=self.p['range_max_m'], min_height=self.p['min_height_m'],
            max_height=self.p['max_height_m'])
        self.tilt = TiltEstimator(
            calibration_sec=self.p['calibration_sec'], tau_sec=self.p['tilt_tau_sec'],
            gravity=self.p['gravity_mps2'], accel_tolerance=self.p['accel_tolerance_mps2'],
            calibration_gyro_limit=self.p['calibration_gyro_limit_rps'],
            calibration_accel_std_limit=self.p['calibration_accel_std_limit'],
            max_gap_sec=self.p['max_imu_gap_sec'])
        self.history = deque(maxlen=600)  # (IMU timestamp, up in cloud frame)
        self.pending_cloud = None
        self.last_imu_wall = None
        self.last_output_wall = None
        self.last_scan_stamp = None
        self.min_input_stamp = 0.0
        self.imu_to_cloud = None
        self.reason = 'WAIT_IMU_AND_TF'
        self.last_stats = {}
        self.counts = dict(clouds_received=0, scans_published=0,
                           invalid_clouds=0, invalid_imus=0, replaced_pending_clouds=0)
        self.buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.listener = TransformListener(self.buffer, self)
        self.broadcaster = TransformBroadcaster(self)
        self.scan_pub = self.create_publisher(LaserScan, self.p['scan_topic'], qos_profile_sensor_data)
        self.cloud_pub = self.create_publisher(
            PointCloud2, self.p['level_cloud_topic'], qos_profile_sensor_data)
        self.status_pub = self.create_publisher(String, '/haic/preprocess_status', 10)
        self.create_subscription(Imu, self.p['imu_topic'], self.on_imu, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, self.p['cloud_topic'], self.on_cloud, qos_profile_sensor_data)
        # Public, volatile notification: do not subscribe to TA status topics.
        self.create_subscription(Empty, '/g1/respawned', self.on_respawn, 10)
        self.wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.02, self.process_pending, clock=self.wall_clock)
        self.create_timer(1.0, self.publish_status, clock=self.wall_clock)
        self.get_logger().info(
            'Stage 1: hold G1 still for initial tilt calibration. '
            'Outputs: /scan and /haic/lidar/points_level. No motor commands, odom or map.')
        if not self.get_parameter('use_sim_time').value:
            self.get_logger().warning('Set use_sim_time:=true for the HAIC simulator.')

    def reset(self, reason):
        self.tilt.reset()
        self.history.clear()
        self.pending_cloud = None
        self.last_imu_wall = None
        self.last_output_wall = None
        self.last_scan_stamp = None
        self.last_stats = {}
        self.reason = reason

    def on_respawn(self, _msg):
        self.min_input_stamp = self.get_clock().now().nanoseconds * 1e-9
        self.reset('RESPAWN_RECALIBRATING')

    def on_imu(self, msg):
        if frame_name(msg.header.frame_id) != self.imu_frame:
            self.reason = 'UNEXPECTED_IMU_FRAME'
            self.counts['invalid_imus'] += 1
            return
        if self.imu_to_cloud is None:
            try:
                t = self.buffer.lookup_transform(self.cloud_frame, self.imu_frame, Time())
                tr, q = t.transform.translation, t.transform.rotation
                if np.linalg.norm([tr.x, tr.y, tr.z]) > 0.02:
                    self.reason = 'IMU_NOT_COLOCATED_WITH_LIDAR'
                    return
                self.imu_to_cloud = quaternion_matrix([q.x, q.y, q.z, q.w])
            except (TransformException, ValueError):
                self.reason = 'WAIT_IMU_TO_LIDAR_TF'
                return
        stamp = stamp_sec(msg.header.stamp)
        now_sim = self.get_clock().now().nanoseconds * 1e-9
        if now_sim < self.min_input_stamp - .1:
            self.min_input_stamp = now_sim  # actual simulation clock reset
            self.reset('CLOCK_RESET_RECALIBRATING')
        if stamp < self.min_input_stamp - 1e-9:
            return
        old_stamp = self.tilt.last_stamp
        if old_stamp is not None and (stamp < old_stamp or stamp-old_stamp > self.p['max_imu_gap_sec']):
            self.reset('IMU_TIME_RESET_RECALIBRATING')
        gyro = msg.angular_velocity
        accel = msg.linear_acceleration
        try:
            up = self.tilt.update(stamp,
                self.imu_to_cloud @ np.array([gyro.x, gyro.y, gyro.z]),
                self.imu_to_cloud @ np.array([accel.x, accel.y, accel.z]))
        except ValueError:
            self.counts['invalid_imus'] += 1
            self.reason = 'INVALID_IMU'
            return
        if old_stamp is not None and stamp == old_stamp:
            return
        self.last_imu_wall = time.monotonic()
        if up is not None:
            self.history.append((stamp, up))
        else:
            self.reason = self.tilt.state

    def on_cloud(self, msg):
        self.counts['clouds_received'] += 1
        if self.pending_cloud is not None:
            self.counts['replaced_pending_clouds'] += 1
        self.pending_cloud = (msg, time.monotonic())

    def process_pending(self):
        if self.pending_cloud is None:
            return
        msg, received = self.pending_cloud
        if frame_name(msg.header.frame_id) != self.cloud_frame:
            self.reason = 'UNEXPECTED_CLOUD_FRAME'
            self.counts['invalid_clouds'] += 1
            self.pending_cloud = None
            return
        now = time.monotonic()
        if now-received > self.p['max_wall_input_age_sec']:
            self.reason = 'STALE_PENDING_CLOUD'
            self.pending_cloud = None
            return
        if self.last_imu_wall is None:
            return  # retain the actionable frame/TF/calibration reason
        if now-self.last_imu_wall > self.p['max_wall_input_age_sec']:
            self.reason = 'WAIT_FRESH_IMU'
            return
        if not self.history:
            self.reason = self.tilt.state
            return
        stamp = stamp_sec(msg.header.stamp)
        # Choose a sample no later than the scan. Never apply an arbitrary
        # newest tilt from the future to an old cloud.
        sample = next((s for s in reversed(self.history) if s[0] <= stamp+1e-9), None)
        if sample is None:
            self.reason = 'WAIT_CALIBRATED_IMU_AT_CLOUD_TIME'
            return
        skew = max(0.0, stamp-sample[0])
        if skew > self.p['max_imu_cloud_skew_sec']:
            self.reason = 'IMU_CLOUD_TIME_SKEW'
            # The IMU may catch up; the bounded pending-cloud lifetime applies.
            return
        up = sample[1]
        tilt_deg = math.degrees(math.acos(float(np.clip(up[2],-1.,1.))))
        if tilt_deg > self.p['max_tilt_deg']:
            self.reason = 'EXCESSIVE_TILT_CHECK_ROBOT'
            self.pending_cloud = None
            return
        try:
            raw = read_xyz(msg)
            rotation = level_rotation(up)
            points = raw[np.all(np.isfinite(raw),axis=1)] @ rotation.T
            distance = np.linalg.norm(points,axis=1)
            points = points[(distance >= self.scan_config.range_min)
                            & (distance <= self.scan_config.range_max)]
            ranges, used = make_scan(points, self.scan_config)
        except (ValueError, TypeError, BufferError, OverflowError) as exc:
            self.reason = 'INVALID_CLOUD: ' + str(exc)
            self.counts['invalid_clouds'] += 1
            self.pending_cloud = None
            return
        self.pending_cloud = None
        if used == 0:
            self.reason = 'NO_POINTS_IN_HEIGHT_BAND'
            return
        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = self.cloud_frame
        transform.child_frame_id = self.level_frame
        # TF stores the child pose in its parent; point conversion uses its inverse.
        q = matrix_quaternion(rotation.T)
        transform.transform.rotation.x = float(q[0])
        transform.transform.rotation.y = float(q[1])
        transform.transform.rotation.z = float(q[2])
        transform.transform.rotation.w = float(q[3])
        self.broadcaster.sendTransform(transform)
        leveled = PointCloud2()
        leveled.header.stamp = msg.header.stamp
        leveled.header.frame_id = self.level_frame
        leveled.height, leveled.width = 1, len(points)
        leveled.fields = [PointField(name=n, offset=i*4, datatype=PointField.FLOAT32, count=1)
                          for i,n in enumerate(('x','y','z'))]
        leveled.is_bigendian = False
        leveled.point_step, leveled.row_step = 12, 12*len(points)
        leveled.is_dense = True
        leveled.data = array('B', np.asarray(points,dtype='<f4').tobytes())
        self.cloud_pub.publish(leveled)
        scan = LaserScan()
        scan.header = leveled.header
        scan.angle_min = -math.pi
        scan.angle_increment = 2*math.pi/self.scan_config.bins
        scan.angle_max = scan.angle_min + (self.scan_config.bins-1)*scan.angle_increment
        scan.range_min = self.scan_config.range_min
        scan.range_max = self.scan_config.range_max
        dt = 0.1 if self.last_scan_stamp is None else stamp-self.last_scan_stamp
        scan.scan_time = float(dt if 0 < dt < 1 else 0.1)
        # HAIC cloud is an instantaneous raycast with no point-time field.
        scan.time_increment = 0.0
        scan.ranges = ranges.tolist()
        self.scan_pub.publish(scan)
        self.last_scan_stamp = stamp
        self.last_output_wall = time.monotonic()
        self.reason = 'PREPROCESS_READY'
        self.counts['scans_published'] += 1
        finite = ranges[np.isfinite(ranges)]
        self.last_stats = dict(raw_points=len(raw), level_points=len(points),
            scan_points=used, occupied_bins=int(len(finite)), bins=len(ranges),
            coverage_percent=round(100*len(finite)/len(ranges),1),
            nearest_range_m=float(np.min(finite)) if len(finite) else None,
            tilt_deg=round(tilt_deg,2), imu_cloud_skew_sec=round(skew,4))

    def publish_status(self):
        now = time.monotonic()
        age = None if self.last_output_wall is None else now-self.last_output_wall
        ready = age is not None and age < self.p['max_wall_input_age_sec'] and self.reason == 'PREPROCESS_READY'
        reason = self.reason
        if self.reason == 'PREPROCESS_READY' and not ready:
            reason = 'STALE_SCAN_OR_SIM_PAUSED'
        payload = dict(stage=1, ready=bool(ready), state=reason,
                       tilt_initialized=self.tilt.up is not None,
                       output_wall_age_sec=None if age is None else round(age,3),
                       absolute_yaw_available=False, localization_available=False,
                       **self.counts, **self.last_stats)
        msg = String()
        msg.data = json.dumps(payload,allow_nan=False)
        self.status_pub.publish(msg)
        if not ready:
            self.get_logger().warning(reason, throttle_duration_sec=5.0)


def main(args=None):
    from .runtime import run_node
    run_node(SensorPreprocess, args)
