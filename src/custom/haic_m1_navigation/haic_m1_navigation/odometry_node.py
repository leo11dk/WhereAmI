"""ROS 2 scan-matching odometry. No simulator state or command integration."""
from __future__ import annotations
from collections import deque
import json
import math
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from .geometry import classify_scan_stamp, matrix_pose, pelvis_from_scan_pose, scan_points
from .icp import register_scan


def _seconds(stamp):
    return stamp.sec + stamp.nanosec*1e-9


class LidarOdometry(Node):
    def __init__(self):
        super().__init__('haic_lidar_odometry')
        params = dict(scan_topic='/scan', odom_frame='odom', base_frame='pelvis',
            max_correspondence_m=.6, max_rmse_m=.18, min_inliers=20,
            min_normal_eigen_ratio=.01, max_scan_gap_sec=.8,
            max_translation_speed_mps=1.0, max_rotation_speed_rps=1.5,
            scan_max_points=600, max_pending_wall_sec=2.0,
            stale_sim_sec=.5, stale_wall_sec=3.0)
        for key,value in params.items():
            self.declare_parameter(key,value)
        self.p = {key:self.get_parameter(key).value for key in params}
        if self.p['odom_frame'] == self.p['base_frame']:
            raise ValueError('odom_frame and base_frame must differ')
        self.buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.listener = TransformListener(self.buffer,self)
        self.broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry,'/odom',10)
        self.status_pub = self.create_publisher(String,'/haic/odometry_status',10)
        self.create_subscription(LaserScan,self.p['scan_topic'],self.on_scan,qos_profile_sensor_data)
        self.create_subscription(Empty,'/g1/respawned',self.on_respawn,10)
        self.session = uuid.uuid4().hex
        self.epoch_count = -1
        self.reset('WAIT_SCAN')
        self.wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(.02,self.process,clock=self.wall_clock)
        self.create_timer(.5,self.publish_status,clock=self.wall_clock)
        self.get_logger().info('LiDAR-only relative odometry: scan origin starts at zero; global XY/yaw are not known.')

    def reset(self,reason,epoch_start_sim_sec=None):
        self.epoch_count += 1
        self.epoch = self.session+':'+str(self.epoch_count)
        self.epoch_reason = reason
        self.epoch_start_sim_sec = (self.get_clock().now().nanoseconds*1e-9
            if epoch_start_sim_sec is None else float(epoch_start_sim_sec))
        self.pending = deque(maxlen=4)
        self.previous_points = None
        self.previous_stamp = None
        self.previous_pelvis = None
        self.odom_from_scan = np.eye(3)
        self.last_input_stamp = None
        self.last_accepted_wall = None
        self.last_accepted_stamp = None
        self.ready = False
        self.state = reason
        self.result = None
        self.publish_status()

    def on_respawn(self,_msg):
        # Public Empty carries no simulator state. Fence with our ROS clock.
        self.reset('RESPAWN_NEW_ODOM_EPOCH')

    def on_scan(self,msg):
        stamp = _seconds(msg.header.stamp)
        decision = classify_scan_stamp(stamp,self.last_input_stamp,
            self.epoch_start_sim_sec,self.get_clock().now().nanoseconds*1e-9)
        if decision == 'clock_reset':
            self.reset('SCAN_CLOCK_RESET',stamp)
        elif decision != 'accept':
            if decision in ('invalid','wait_clock'):
                self.ready,self.state = False,'WAIT_VALID_SCAN_CLOCK'
            return
        self.last_input_stamp = stamp
        self.pending.append((msg,time.monotonic()))

    def process(self):
        if not self.pending:
            return
        msg,received = self.pending[0]
        if time.monotonic()-received > self.p['max_pending_wall_sec']:
            self.pending.popleft()
            self.ready,self.state = False,'STALE_PENDING_SCAN'
            return
        stamp = _seconds(msg.header.stamp)
        if stamp < self.epoch_start_sim_sec-1e-9:
            self.pending.popleft()
            return
        try:
            tf = self.buffer.lookup_transform(self.p['base_frame'],msg.header.frame_id,
                                             Time.from_msg(msg.header.stamp))
        except TransformException:
            self.ready,self.state = False,'WAIT_SCAN_TIME_EXTRINSIC_TF'
            return
        self.pending.popleft()
        try:
            points = scan_points(msg,self.p['scan_max_points'])
            q,t = tf.transform.rotation,tf.transform.translation
            extrinsic = np.eye(4)
            extrinsic[:3,:3] = Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
            extrinsic[:3,3] = [t.x,t.y,t.z]
        except (ValueError,TypeError):
            self.ready,self.state = False,'INVALID_SCAN_OR_EXTRINSIC'
            return
        if len(points) < self.p['min_inliers']:
            self.ready,self.state = False,'INSUFFICIENT_SCAN_POINTS'
            return
        if self.previous_stamp is not None and stamp-self.previous_stamp > self.p['max_scan_gap_sec']:
            # A fresh local origin cannot silently continue an old saved map.
            self.reset('TRACKING_GAP_NEW_ODOM_EPOCH',stamp)
        if self.previous_points is None:
            self.previous_points,self.previous_stamp = points,stamp
            pelvis = pelvis_from_scan_pose(matrix_pose(self.odom_from_scan),extrinsic)
            self.previous_pelvis = pelvis
            self.publish_odometry(msg.header.stamp,pelvis,None,None)
            self.last_accepted_wall,self.last_accepted_stamp = time.monotonic(),stamp
            self.ready,self.state = False,'INITIALIZED_WAIT_SECOND_SCAN'
            self.publish_status()
            return
        dt = stamp-self.previous_stamp
        if dt <= 0:
            self.ready,self.state = False,'NONINCREASING_SCAN_TIME'
            return
        match = register_scan(points,self.previous_points,
            max_correspondence_m=self.p['max_correspondence_m'],
            max_rmse_m=self.p['max_rmse_m'],min_inliers=self.p['min_inliers'],
            min_normal_eigen_ratio=self.p['min_normal_eigen_ratio'])
        self.result = match
        if not match.accepted:
            self.ready,self.state = False,match.reason
            self.publish_status()
            return
        delta = matrix_pose(match.transform)
        if (np.linalg.norm(delta[:2]) > self.p['max_translation_speed_mps']*dt+.03
                or abs(delta[2]) > self.p['max_rotation_speed_rps']*dt+.03):
            self.ready,self.state = False,'IMPLAUSIBLE_SCAN_JUMP'
            self.publish_status()
            return
        candidate = self.odom_from_scan @ match.transform
        pelvis = pelvis_from_scan_pose(matrix_pose(candidate),extrinsic)
        self.publish_odometry(msg.header.stamp,pelvis,self.previous_pelvis,dt)
        self.odom_from_scan = candidate
        self.previous_points,self.previous_stamp,self.previous_pelvis = points,stamp,pelvis
        self.last_accepted_wall,self.last_accepted_stamp = time.monotonic(),stamp
        self.ready,self.state = True,'ODOMETRY_READY'
        self.publish_status()

    def publish_odometry(self,stamp,pelvis,previous,dt):
        q = Rotation.from_matrix(pelvis[:3,:3]).as_quat()
        tf = TransformStamped()
        tf.header.stamp,tf.header.frame_id = stamp,self.p['odom_frame']
        tf.child_frame_id = self.p['base_frame']
        tf.transform.translation.x = float(pelvis[0,3])
        tf.transform.translation.y = float(pelvis[1,3])
        tf.transform.translation.z = 0.0
        tf.transform.rotation.x,tf.transform.rotation.y = float(q[0]),float(q[1])
        tf.transform.rotation.z,tf.transform.rotation.w = float(q[2]),float(q[3])
        msg = Odometry()
        msg.header,msg.child_frame_id = tf.header,tf.child_frame_id
        msg.pose.pose.position.x,msg.pose.pose.position.y = float(pelvis[0,3]),float(pelvis[1,3])
        msg.pose.pose.orientation = tf.transform.rotation
        # Illustrative conservative covariance, not a calibrated uncertainty bound.
        variance = max(.02,(self.result.rmse if self.result else .1))**2
        for index,value in ((0,variance),(7,variance),(14,1e6),(21,.05),(28,.05),(35,.03)):
            msg.pose.covariance[index] = float(value)
        if previous is not None and dt is not None:
            velocity = pelvis[:3,:3].T @ ((pelvis[:3,3]-previous[:3,3])/dt)
            omega_previous = Rotation.from_matrix(previous[:3,:3].T@pelvis[:3,:3]).as_rotvec()/dt
            omega = pelvis[:3,:3].T @ previous[:3,:3] @ omega_previous
            msg.twist.twist.linear.x,msg.twist.twist.linear.y,msg.twist.twist.linear.z = map(float,velocity)
            msg.twist.twist.angular.x,msg.twist.twist.angular.y,msg.twist.twist.angular.z = map(float,omega)
        for index in (0,7,14,21,28,35):
            msg.twist.covariance[index] = .1 if index in (0,7,35) else 1e6
        self.broadcaster.sendTransform(tf)
        self.odom_pub.publish(msg)

    def publish_status(self):
        wall = time.monotonic()
        sim = self.get_clock().now().nanoseconds*1e-9
        fresh = (self.last_accepted_wall is not None
                 and wall-self.last_accepted_wall < self.p['stale_wall_sec']
                 and -.1 <= sim-self.last_accepted_stamp < self.p['stale_sim_sec'])
        ready = bool(self.ready and fresh)
        def finite(value): return float(value) if value is not None and math.isfinite(value) else None
        match = self.result
        payload = dict(ready=ready,state=self.state if fresh or not self.ready else 'STALE_OR_SIM_PAUSED',
            epoch=self.epoch,epoch_start_sim_sec=self.epoch_start_sim_sec,
            epoch_reason=self.epoch_reason,
            stamp_sim_sec=self.last_accepted_stamp,
            residual_m=finite(match.rmse if match else None),inliers=match.inliers if match else 0,
            condition=finite(match.condition if match else None),degenerate=match.degenerate if match else True,
            planar_z_gauge=True,global_position_available=False)
        msg = String();msg.data = json.dumps(payload,allow_nan=False)
        self.status_pub.publish(msg)


def main(args=None):
    from haic_m1_sensors.runtime import run_node
    run_node(LidarOdometry, args)
