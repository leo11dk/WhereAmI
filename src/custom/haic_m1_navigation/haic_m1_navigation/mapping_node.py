"""Chapter 3: build a local 2D map from accepted sensor odometry and scans."""

from collections import deque
import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from .geometry import scan_points, transform_points
from .mapping_core import GridConfig, OccupancyMap, save_map, timestamp_in_epoch


def _stamp(stamp):return stamp.sec+stamp.nanosec*1e-9


def _transform_2d(transform):
    t=transform.transform.translation;q=transform.transform.rotation
    norm=math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)
    if not math.isfinite(norm) or norm<1e-12:raise ValueError('invalid TF quaternion')
    x,y,z,w=q.x/norm,q.y/norm,q.z/norm,q.w/norm
    yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z));c,s=math.cos(yaw),math.sin(yaw)
    return np.array([[c,-s,t.x],[s,c,t.y],[0.,0.,1.]])


def _map_message(grid,frame,stamp):
    msg=OccupancyGrid();msg.header.frame_id=frame;msg.header.stamp=stamp
    c=grid.config
    msg.info.resolution=c.resolution;msg.info.width=c.width;msg.info.height=c.height
    msg.info.origin.position.x=c.origin_x;msg.info.origin.position.y=c.origin_y
    msg.info.origin.orientation.w=1.0
    msg.data=grid.occupancy().ravel().tolist()
    return msg


class MappingNode(Node):
    def __init__(self):
        super().__init__('haic_m1_mapping')
        defaults=dict(data_dir='custom/m1_data',map_frame='map',odom_frame='odom',
                      scan_topic='/scan',resolution=.10,origin_x=-30.,origin_y=-30.,
                      width=600,height=600,max_scan_points=180,max_sensor_age_sec=.5,
                      max_wall_age_sec=3.,minimum_save_scans=5)
        for key,value in defaults.items():self.declare_parameter(key,value)
        self.p={key:self.get_parameter(key).value for key in defaults}
        self.grid=OccupancyMap(GridConfig(self.p['resolution'],self.p['origin_x'],self.p['origin_y'],
                                         self.p['width'],self.p['height']))
        self.buffer=Buffer();self.listener=TransformListener(self.buffer,self)
        self.broadcaster=TransformBroadcaster(self)
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub=self.create_publisher(OccupancyGrid,'/map',qos)
        self.status_pub=self.create_publisher(String,'/haic/mapping_status',10)
        self.create_subscription(LaserScan,self.p['scan_topic'],self.on_scan,qos_profile_sensor_data)
        self.create_subscription(String,'/haic/odometry_status',self.on_odometry,10)
        self.create_service(Trigger,'/haic/save_map',self.on_save)
        self.pending=deque(maxlen=15);self.odometry={};self.odometry_wall=None
        self.epoch=None;self.locked=False;self.reason='WAIT_ODOMETRY';self.accepted_stamp=None
        self.accepted_wall=None;self.last_frame=None;self.initial_reference=None
        self.last_map_wall=-math.inf
        self.wall_clock=Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(.05,self.process,clock=self.wall_clock)
        self.create_timer(1.,self.publish_status,clock=self.wall_clock)
        self.get_logger().info('Mapping only: teleoperate slowly, inspect /map, call /haic/save_map when verified.')

    def on_odometry(self,msg):
        try:data=json.loads(msg.data)
        except (ValueError,TypeError):return
        if not isinstance(data,dict):return
        epoch=data.get('epoch')
        if epoch is not None and epoch!=self.epoch:
            if self.epoch is not None and self.grid.scans:
                self.locked=True;self.reason='ODOMETRY_RESET_START_NEW_MAP'
            self.epoch=epoch;self.pending.clear()
        self.odometry=data;self.odometry_wall=time.monotonic()

    def on_scan(self,msg):self.pending.append((msg,time.monotonic()))

    def odometry_ready(self):
        if not self.odometry.get('ready') or self.odometry_wall is None:return False
        stamp=self.odometry.get('stamp_sim_sec')
        if not timestamp_in_epoch(stamp,self.odometry.get('epoch_start_sim_sec')):return False
        now=self.get_clock().now().nanoseconds*1e-9
        return (time.monotonic()-self.odometry_wall<self.p['max_wall_age_sec']
                and -.1<=now-stamp<self.p['max_sensor_age_sec'])

    def process(self):
        if self.locked:return
        if not self.odometry_ready():
            self.reason='WAIT_FRESH_ACCEPTED_ODOMETRY';return
        if not self.pending:return
        scan,received=self.pending[0]
        if time.monotonic()-received>self.p['max_wall_age_sec']:
            self.pending.popleft();self.reason='STALE_SCAN';return
        stamp=_stamp(scan.header.stamp)
        if not timestamp_in_epoch(stamp,self.odometry.get('epoch_start_sim_sec')):
            self.pending.popleft();self.reason='SCAN_FROM_PREVIOUS_ODOMETRY_EPOCH';return
        now=self.get_clock().now().nanoseconds*1e-9
        if not -.1<=now-stamp<self.p['max_sensor_age_sec']:
            self.pending.popleft();self.reason='SCAN_TIME_MISMATCH_OR_STALE';return
        if self.accepted_stamp is not None and stamp<=self.accepted_stamp:
            self.pending.popleft();return
        try:
            tf=self.buffer.lookup_transform(self.p['odom_frame'],scan.header.frame_id,
                                            Time.from_msg(scan.header.stamp))
        except TransformException:
            self.reason='WAIT_ODOM_SCAN_TF_AT_STAMP';return
        self.pending.popleft()
        try:
            transform=_transform_2d(tf)
            endpoints=transform_points(transform,scan_points(scan,max_points=self.p['max_scan_points']))
            hits=self.grid.update(transform[:2,2],endpoints)
        except (ValueError,TypeError) as exc:
            self.reason='MAP_UPDATE_REJECTED: '+str(exc);return
        if not hits:
            self.reason='NO_VALID_MAP_HITS';return
        if self.initial_reference is None:
            self.initial_reference=dict(scan_frame=scan.header.frame_id,
                scan_origin_map=transform[:2,2].tolist(),stamp_sim_sec=stamp)
        self.accepted_stamp=stamp;self.accepted_wall=time.monotonic();self.last_frame=scan.header.frame_id
        out=TransformStamped();out.header.stamp=scan.header.stamp
        out.header.frame_id=self.p['map_frame'];out.child_frame_id=self.p['odom_frame']
        out.transform.rotation.w=1.0
        self.broadcaster.sendTransform(out)
        if time.monotonic()-self.last_map_wall>=1.0:
            self.map_pub.publish(_map_message(self.grid,self.p['map_frame'],scan.header.stamp))
            self.last_map_wall=time.monotonic()
        self.reason='MAPPING_ACTIVE';self.publish_status()

    def on_save(self,request,response):
        del request
        if self.locked:
            response.success=False;response.message=self.reason;return response
        if (not self.odometry_ready() or self.accepted_wall is None
                or time.monotonic()-self.accepted_wall>=self.p['max_wall_age_sec']
                or self.reason!='MAPPING_ACTIVE'):
            response.success=False;response.message='Save requires fresh accepted odometry and mapping evidence.'
            return response
        try:
            manifest=save_map(self.grid,self.p['data_dir'],
                metadata=dict(odometry_epoch=self.epoch,map_frame=self.p['map_frame'],
                              initial_reference=self.initial_reference),
                min_scans=self.p['minimum_save_scans'])
        except (ValueError,OSError,TypeError) as exc:
            response.success=False;response.message=str(exc);return response
        response.success=True
        response.message=json.dumps(dict(map_id=manifest['map_id'],
            data_dir=str(Path(self.p['data_dir']).expanduser().resolve()),yaml=manifest['yaml']))
        self.get_logger().info('Saved map '+response.message)
        return response

    def publish_status(self):
        age=None if self.accepted_wall is None else time.monotonic()-self.accepted_wall
        ready=(not self.locked and self.odometry_ready() and age is not None
               and age<self.p['max_wall_age_sec'] and self.reason=='MAPPING_ACTIVE')
        data=dict(ready=bool(ready),state=self.reason,epoch=self.epoch,
                  epoch_start_sim_sec=self.odometry.get('epoch_start_sim_sec'),
                  stamp_sim_sec=self.accepted_stamp,scans=self.grid.scans,
                  hit_rays=self.grid.hit_rays,occupied_cells=int(np.count_nonzero(
                      self.grid.observed & (self.grid.log_odds>=math.log(.65/.35)))),
                  data_dir=str(Path(self.p['data_dir']).expanduser().resolve()))
        msg=String();msg.data=json.dumps(data,allow_nan=False);self.status_pub.publish(msg)


def main(args=None):
    from haic_m1_sensors.runtime import run_node
    run_node(MappingNode, args)
