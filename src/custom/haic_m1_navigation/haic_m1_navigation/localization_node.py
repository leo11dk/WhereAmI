"""Chapter 4: fixed-map scan matching, requiring an explicit map-frame seed."""

from collections import deque
import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from .geometry import matrix_pose, scan_points, se2_matrix, wrap_angle
from .icp import register_points
from .mapping_core import load_map, timestamp_in_epoch
from .mapping_node import _map_message, _stamp, _transform_2d
from .recovery import RecoveryGate, spawn_in_map


class LocalizationNode(Node):
    def __init__(self):
        super().__init__('haic_m1_localization')
        defaults=dict(data_dir='custom/m1_data',map_frame='map',odom_frame='odom',
            pelvis_frame='pelvis',scan_topic='/scan',max_scan_points=240,
            max_sensor_age_sec=.5,max_wall_age_sec=3.,max_correspondence_m=.8,
            max_rmse_m=.20,min_inliers=35,min_normal_eigen_ratio=.01,
            min_inlier_fraction=.35,max_correction_translation_m=.6,max_correction_yaw_deg=20.,
            auto_seed_on_start=False)
        for key,value in defaults.items():self.declare_parameter(key,value)
        self.p={key:self.get_parameter(key).value for key in defaults}
        self.grid,self.metadata=load_map(self.p['data_dir'])
        self.map_id=self.metadata['map_id'];self.target=self.grid.occupied_points()
        if len(self.target)<self.p['min_inliers']:
            raise ValueError('saved map has too few occupied cells; remap with more structure')
        self.buffer=Buffer();self.listener=TransformListener(self.buffer,self)
        self.broadcaster=TransformBroadcaster(self)
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub=self.create_publisher(OccupancyGrid,'/map',qos)
        self.status_pub=self.create_publisher(String,'/haic/localization_status',10)
        self.create_subscription(LaserScan,self.p['scan_topic'],self.on_scan,qos_profile_sensor_data)
        self.create_subscription(String,'/haic/odometry_status',self.on_odometry,10)
        self.create_subscription(PoseWithCovarianceStamped,'/initialpose',self.on_seed,10)
        self.create_subscription(Empty,'/g1/respawned',self.on_respawn,10)
        self.recovery=RecoveryGate(self.p['auto_seed_on_start'])
        self.pending=deque(maxlen=15);self.odometry={};self.odometry_wall=None
        self.epoch=None;self.map_T_odom=None;self.pending_seed=None
        self.accepted_stamp=None;self.accepted_wall=None;self.quality={}
        self.reason='WAIT_INITIALPOSE_IN_MAP';self.last_scan_stamp=None
        self.wall_clock=Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(.05,self.process,clock=self.wall_clock)
        self.create_timer(.5,self.publish_status,clock=self.wall_clock)
        self.map_pub.publish(_map_message(self.grid,self.p['map_frame'],self.get_clock().now().to_msg()))
        self.get_logger().info('Loaded '+self.map_id+'. In RViz map frame, use 2D Pose Estimate for pelvis pose.')

    def on_odometry(self,msg):
        try:data=json.loads(msg.data)
        except (ValueError,TypeError):return
        if not isinstance(data,dict):return
        epoch=data.get('epoch')
        if epoch is not None and epoch!=self.epoch:
            if self.epoch is not None:
                self.map_T_odom=None;self.pending_seed=None;self.accepted_stamp=None
                self.accepted_wall=None;self.last_scan_stamp=None;self.quality={}
                self.reason='ODOMETRY_RESET_SET_INITIALPOSE_AGAIN'
            self.epoch=epoch;self.pending.clear()
        self.odometry=data;self.odometry_wall=time.monotonic()

    def on_scan(self,msg):self.pending.append((msg,time.monotonic()))

    def on_respawn(self,_msg):
        self.recovery.respawn(self.get_clock().now().nanoseconds*1e-9)
        self.map_T_odom=None;self.pending_seed=None
        self.accepted_stamp=None;self.accepted_wall=None;self.quality={}
        self.pending.clear();self.last_scan_stamp=None
        self.reason='RESPAWN_WAIT_NEW_ODOMETRY'

    def try_known_spawn_seed(self):
        if not self.recovery.eligible(self.odometry):return
        try:
            data=json.loads((Path(self.p['data_dir'])/'world_alignment.json').read_text())
            value=spawn_in_map(data,self.map_id)
        except (OSError,ValueError,KeyError,TypeError):
            self.reason='AUTO_SEED_NEEDS_VERIFIED_WORLD_ALIGNMENT';return
        if not self.grid.inside(*self.grid.cell(value[:2])):
            self.reason='KNOWN_SPAWN_OUTSIDE_MAP';return
        self.pending_seed=(value,Time().to_msg(),self.epoch,time.monotonic())
        self.recovery.consume(self.epoch)
        self.reason='KNOWN_SPAWN_SEED_WAIT_SENSOR_MATCH'

    def odometry_ready(self):
        if not self.odometry.get('ready') or self.odometry_wall is None:return False
        stamp=self.odometry.get('stamp_sim_sec')
        if not timestamp_in_epoch(stamp,self.odometry.get('epoch_start_sim_sec')):return False
        now=self.get_clock().now().nanoseconds*1e-9
        return (time.monotonic()-self.odometry_wall<self.p['max_wall_age_sec']
                and -.1<=now-stamp<self.p['max_sensor_age_sec'])

    def on_seed(self,msg):
        if msg.header.frame_id.lstrip('/')!=self.p['map_frame']:
            self.reason='INITIALPOSE_MUST_BE_IN_MAP';return
        if not self.odometry_ready():
            self.reason='WAIT_ODOMETRY_THEN_RESEND_INITIALPOSE';return
        seed_stamp=_stamp(msg.header.stamp)
        if seed_stamp!=0 and not timestamp_in_epoch(seed_stamp,self.odometry.get('epoch_start_sim_sec')):
            self.reason='INITIALPOSE_FROM_PREVIOUS_ODOMETRY_EPOCH';return
        pose=msg.pose.pose;q=pose.orientation
        norm=math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)
        if not math.isfinite(norm) or norm<1e-12:
            self.reason='INVALID_INITIALPOSE';return
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y)/norm**2,1-2*(q.y*q.y+q.z*q.z)/norm**2)
        value=np.array([pose.position.x,pose.position.y,yaw])
        if not np.all(np.isfinite(value)):
            self.reason='INVALID_INITIALPOSE';return
        if not self.grid.inside(*self.grid.cell(value[:2])):
            self.reason='INITIALPOSE_OUTSIDE_MAP';return
        # Record the epoch; a later reset invalidates this request as well.
        self.pending_seed=(value,msg.header.stamp,self.epoch,time.monotonic())
        self.recovery.cancel()  # explicit practice initialization takes precedence
        self.map_T_odom=None;self.accepted_stamp=None;self.accepted_wall=None
        self.reason='WAIT_INITIALPOSE_TIMESTAMP_TF'

    def apply_pending_seed(self):
        if self.pending_seed is None:return
        value,stamp,epoch,received=self.pending_seed
        if epoch!=self.epoch or time.monotonic()-received>self.p['max_wall_age_sec']:
            self.pending_seed=None;self.reason='INITIALPOSE_EXPIRED_RESEND';return
        try:
            lookup=Time() if _stamp(stamp)==0 else Time.from_msg(stamp)
            tf=self.buffer.lookup_transform(self.p['odom_frame'],self.p['pelvis_frame'],lookup)
        except TransformException:return
        # In particular, a zero-stamp seed must not use buffered old-epoch TF
        # before the current epoch's accepted transform reaches this listener.
        if not timestamp_in_epoch(_stamp(tf.header.stamp),self.odometry.get('epoch_start_sim_sec')):
            self.reason='WAIT_CURRENT_EPOCH_PELVIS_TF';return
        odom_T_pelvis=_transform_2d(tf)
        self.map_T_odom=se2_matrix(value)@np.linalg.inv(odom_T_pelvis)
        self.pending_seed=None;self.pending.clear();self.reason='SEED_SET_WAIT_MAP_MATCH'

    def process(self):
        if not self.odometry_ready():
            self.reason='WAIT_FRESH_ACCEPTED_ODOMETRY';return
        self.try_known_spawn_seed()
        self.apply_pending_seed()
        if self.map_T_odom is None:
            if self.pending_seed is None and self.reason=='WAIT_FRESH_ACCEPTED_ODOMETRY':
                self.reason='WAIT_INITIALPOSE_IN_MAP'
            return
        if not self.pending:return
        scan,received=self.pending[0]
        if time.monotonic()-received>self.p['max_wall_age_sec']:
            self.pending.popleft();self.reason='STALE_SCAN';return
        stamp=_stamp(scan.header.stamp);now=self.get_clock().now().nanoseconds*1e-9
        if not timestamp_in_epoch(stamp,self.odometry.get('epoch_start_sim_sec')):
            self.pending.popleft();self.reason='SCAN_FROM_PREVIOUS_ODOMETRY_EPOCH';return
        if not -.1<=now-stamp<self.p['max_sensor_age_sec']:
            self.pending.popleft();self.reason='SCAN_TIME_MISMATCH_OR_STALE';return
        if self.last_scan_stamp is not None and stamp<=self.last_scan_stamp:
            self.pending.popleft();return
        try:
            odom_T_scan=_transform_2d(self.buffer.lookup_transform(
                self.p['odom_frame'],scan.header.frame_id,Time.from_msg(scan.header.stamp)))
        except TransformException:
            self.reason='WAIT_ODOM_SCAN_TF_AT_STAMP';return
        self.pending.popleft();self.last_scan_stamp=stamp
        points=scan_points(scan,max_points=self.p['max_scan_points'])
        guess=self.map_T_odom@odom_T_scan
        try:
            match=register_points(points,self.target,initial=guess,
                max_correspondence_m=self.p['max_correspondence_m'],
                max_rmse_m=self.p['max_rmse_m'],min_inliers=self.p['min_inliers'],
                min_normal_eigen_ratio=self.p['min_normal_eigen_ratio'],
                min_inlier_fraction=self.p['min_inlier_fraction'])
        except (ValueError,TypeError,np.linalg.LinAlgError) as exc:
            self.reason='MAP_MATCH_ERROR: '+str(exc);self.quality={};return
        self.quality=dict(residual_m=float(match.rmse) if math.isfinite(match.rmse) else None,
            inliers=match.inliers,condition=float(match.condition) if math.isfinite(match.condition) else None,
            degenerate=bool(match.degenerate),iterations=match.iterations)
        if not match.accepted:
            self.reason='MAP_MATCH_REJECTED: '+match.reason;return
        correction_xy=match.transform[:2,2]-guess[:2,2]
        correction_yaw=wrap_angle(matrix_pose(match.transform)[2]-matrix_pose(guess)[2])
        if (np.linalg.norm(correction_xy)>self.p['max_correction_translation_m']
                or abs(correction_yaw)>math.radians(self.p['max_correction_yaw_deg'])):
            self.reason='MAP_CORRECTION_TOO_LARGE_RESEED';return
        self.map_T_odom=match.transform@np.linalg.inv(odom_T_scan)
        x,y,yaw=matrix_pose(self.map_T_odom)
        out=TransformStamped();out.header.stamp=scan.header.stamp
        out.header.frame_id=self.p['map_frame'];out.child_frame_id=self.p['odom_frame']
        out.transform.translation.x=float(x);out.transform.translation.y=float(y)
        out.transform.rotation.z=math.sin(yaw/2);out.transform.rotation.w=math.cos(yaw/2)
        self.broadcaster.sendTransform(out)
        self.accepted_stamp=stamp;self.accepted_wall=time.monotonic();self.reason='LOCALIZED'
        self.publish_status()

    def publish_status(self):
        now=self.get_clock().now().nanoseconds*1e-9
        fresh=(self.accepted_stamp is not None and self.accepted_wall is not None
            and -.1<=now-self.accepted_stamp<self.p['max_sensor_age_sec']
            and time.monotonic()-self.accepted_wall<self.p['max_wall_age_sec'])
        ready=fresh and self.odometry_ready() and self.reason=='LOCALIZED'
        data=dict(ready=bool(ready),state=self.reason,map_id=self.map_id,epoch=self.epoch,
                  epoch_start_sim_sec=self.odometry.get('epoch_start_sim_sec'),
                  stamp_sim_sec=self.accepted_stamp,**self.quality)
        msg=String();msg.data=json.dumps(data,allow_nan=False);self.status_pub.publish(msg)


def main(args=None):
    from haic_m1_sensors.runtime import run_node
    run_node(LocalizationNode, args)
