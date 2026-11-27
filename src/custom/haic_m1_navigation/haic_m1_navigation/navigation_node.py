"""Chapter 5: gated, conservative map-goal following with live obstacle stops.

Actual robot dimensions and LiDAR-to-floor height MUST be measured first.
No motor commands are sent while idle. Teleop and this navigator must not own
/cmd_vel simultaneously. This is a small prototype, not a validated safety system.
"""
from __future__ import annotations

from collections import OrderedDict, deque
import json
import math
import time
import uuid

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformException, TransformListener

from haic_m1_sensors.core import quaternion_matrix, read_xyz
from .configuration import NAVIGATION_DEFAULTS
from .ground import floor_height
from .recovery import RecoveryGate
from .planning import (
    ArrivalGate, GridGeometry, limit_command, plan_path, plan_observed_path, segment_free,
    swept_collision, waypoint_command,
)


def stamp_seconds(stamp):return stamp.sec+stamp.nanosec*1e-9


class Navigator(Node):
    def __init__(self):
        super().__init__('haic_m1_navigation')
        for k,v in NAVIGATION_DEFAULTS.items():self.declare_parameter(k,v)
        self.buffer=Buffer();self.listener=TransformListener(self.buffer,self)
        self.occupancy=None;self.geom=None;self.map_error='WAIT_MAP'
        self.map_dirty=False;self.observed_floor=None
        self.last_pose=None
        self.health_lost_wall=None
        self.inputs={};self.observations={};self.processed={}
        self.observation_queue={key:deque(maxlen=20) for key in ('scan','cloud')}
        self.localization={};self.odometry_status={}
        self.localization_wall=self.odom_status_wall=None
        self.recovery=RecoveryGate()
        self.state='IDLE';self.reason='WAIT_GOAL';self.goal_id=None;self.goal=None
        self.goal_epoch=None;self.started=None;self.path=[];self.path_index=0
        self.free=None;self.last_command=(0.,0.);self.last_tick=None
        self.arrival=None;self.blocked_since=None;self.last_progress=None;self.best_distance=None
        self.waypoint_best=None
        self.completed=OrderedDict();self.fingerprints={}
        self.cmd_pub=self.create_publisher(Twist,'/cmd_vel',10)
        path_qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.path_pub=self.create_publisher(Path,'/haic/navigation_path',path_qos)
        status_qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub=self.create_publisher(String,'/haic/navigation_status',status_qos)
        self.create_subscription(OccupancyGrid,'/map',self.on_map,status_qos)
        self.create_subscription(LaserScan,'/scan',lambda m:self.observe('scan',m),qos_profile_sensor_data)
        self.create_subscription(PointCloud2,'/haic/lidar/points_level',lambda m:self.observe('cloud',m),qos_profile_sensor_data)
        self.create_subscription(Odometry,'/odom',lambda m:self.observe('odom',m),qos_profile_sensor_data)
        self.create_subscription(String,'/haic/localization_status',lambda m:self.on_json('localization',m),10)
        self.create_subscription(String,'/haic/odometry_status',lambda m:self.on_json('odometry_status',m),10)
        self.create_subscription(String,'/haic/navigation_goal',self.on_goal,10)
        self.create_subscription(String,'/haic/navigation_cancel',self.on_cancel,10)
        self.create_subscription(Empty,'/g1/respawned',self.on_respawn,10)
        self.create_subscription(PoseStamped,'/goal_pose',self.on_manual_goal,10)
        self.create_timer(.05,self.control)
        self.wall_clock=Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(.5,self.wall_tick,clock=self.wall_clock)
        self.get_logger().info('Navigator loaded; waiting for fresh mapping, pose and obstacle observations.')

    def p(self,name):return self.get_parameter(name).value
    def now(self):return self.get_clock().now().nanoseconds*1e-9

    def on_map(self,msg):
        try:
            if msg.header.frame_id.lstrip('/')!='map':raise ValueError('MAP_FRAME_MUST_BE_MAP')
            if not 0 < msg.info.width*msg.info.height <= 4000000:raise ValueError('MAP_SIZE_INVALID')
            q=msg.info.origin.orientation;r=quaternion_matrix([q.x,q.y,q.z,q.w])
            if abs(r[2,0])+abs(r[2,1])+abs(r[2,2]-1)>.01:raise ValueError('MAP_ORIGIN_NOT_PLANAR')
            occupancy=np.asarray(msg.data,dtype=np.int16).reshape(msg.info.height,msg.info.width)
            geom=GridGeometry(msg.info.resolution,msg.info.origin.position.x,
                              msg.info.origin.position.y,math.atan2(r[1,0],r[0,0]))
        except (ValueError,TypeError) as exc:
            self.map_error=str(exc);return
        changed=(self.occupancy is not None and
                 (self.geom!=geom or not np.array_equal(self.occupancy,occupancy)))
        self.occupancy,self.geom=occupancy,geom;self.map_error=None
        if changed and self.state=='ACTIVE':
            if self.p('online_mapping'):self.map_dirty=True
            else:self.finish('FAILED','MAP_CHANGED_REPLAN_REQUIRED')

    def observe(self,key,msg):
        value=(msg,time.monotonic());self.inputs[key]=value
        if key in self.observation_queue:self.observation_queue[key].append(value)

    def on_json(self,key,msg):
        try:value=json.loads(msg.data)
        except (ValueError,TypeError):return
        if not isinstance(value,dict):return
        setattr(self,key,value)
        setattr(self,'localization_wall' if key=='localization' else 'odom_status_wall',time.monotonic())

    def on_respawn(self,_msg):
        self.recovery.respawn(self.now())
        if self.state=='ACTIVE':self.finish('FAILED','PUBLIC_RESPAWN_REQUIRES_NEW_GOAL')
        self.inputs.clear();self.observations.clear();self.processed.clear()
        for queue in self.observation_queue.values():queue.clear()

    def fresh(self,stamp,wall,now):
        try:age=now-float(stamp)
        except (TypeError,ValueError):return False
        return (-.1<=age<self.p('input_timeout_sec') and wall is not None
                and time.monotonic()-wall<self.p('max_wall_age_sec'))

    def transform(self,frame,stamp):
        t=self.buffer.lookup_transform('map',frame,Time.from_msg(stamp))
        q=t.transform.rotation;v=t.transform.translation
        return quaternion_matrix([q.x,q.y,q.z,q.w]),np.array([v.x,v.y,v.z])

    def pose(self):
        t=self.buffer.lookup_transform('map','pelvis',Time())
        q=t.transform.rotation;r=quaternion_matrix([q.x,q.y,q.z,q.w]);v=t.transform.translation
        if not -.1<=self.now()-stamp_seconds(t.header.stamp)<self.p('max_pose_age_sec'):
            raise ValueError('STALE_POSE_TF')
        return np.array([v.x,v.y,math.atan2(r[1,0],r[0,0])])

    def process_observations(self):
        for key in ('scan','cloud'):
            value=None
            # TF arrives after cloud/scan on independent DDS subscriptions.
            # Keep recent inputs so continuously arriving newer clouds cannot
            # starve the last scan whose timestamped TF is already available.
            for candidate in reversed(self.observation_queue[key]):
                msg,wall=candidate;stamp=stamp_seconds(msg.header.stamp)
                if stamp<=self.processed.get(key,-math.inf):continue
                if not self.fresh(stamp,wall,self.now()):continue
                try:rotation,translation=self.transform(msg.header.frame_id,msg.header.stamp)
                except TransformException:continue
                value=candidate;break
            if value is None:continue
            try:
                if key=='scan':
                    ranges=np.asarray(msg.ranges,dtype=float)
                    angle=msg.angle_min+np.arange(len(ranges))*msg.angle_increment
                    valid=np.isfinite(ranges)&(ranges>=msg.range_min)&(ranges<=msg.range_max)
                    if valid.sum()<self.p('min_scan_hits'):raise ValueError('INSUFFICIENT_SCAN_HITS')
                    points=np.column_stack((ranges[valid]*np.cos(angle[valid]),
                                            ranges[valid]*np.sin(angle[valid]),np.zeros(valid.sum())))
                else:
                    points=read_xyz(msg)
                    if len(points)>self.p('cloud_point_budget'):raise ValueError('POINT_BUDGET_EXCEEDED')
                    floor=self.p('lidar_floor_z_m')
                    if self.p('online_mapping'):
                        floor=floor_height(points)
                        self.observed_floor=floor
                    height=points[:,2]-floor
                    keep=(np.all(np.isfinite(points),axis=1)&(height>=self.p('obstacle_min_height_m'))
                          &(height<=self.p('obstacle_max_height_m')))
                    keep &= np.hypot(points[:,0],points[:,1]) > self.p('robot_radius_m')
                    points=points[keep]
                points=(points@rotation.T+translation)[:,:2]
                # Retain one actual point in each occupied 5cm XY cell. Collision
                # envelope below includes the cell diagonal as extra clearance.
                if len(points):
                    _,ix=np.unique(np.floor(points/.05).astype(np.int64),axis=0,return_index=True)
                    points=points[ix]
                self.observations[key]=(points,stamp,wall,None)
                self.processed[key]=stamp
            except TransformException:
                # Same input can be retried once timestamped TF arrives.
                continue
            except (ValueError,TypeError,BufferError,OverflowError) as exc:
                self.observations[key]=(np.empty((0,2)),stamp,wall,str(exc))
                self.processed[key]=stamp

    def readiness(self):
        flags=('allow_motion',) if self.p('online_mapping') else (
            'allow_motion','footprint_verified','route_verified','ground_verified')
        for name in flags:
            if self.p(name) is not True:return False,'SET_'+name.upper()+'_AFTER_MEASUREMENT'
        if self.p('online_mapping'):
            if self.localization.get('source')!='documented_spawn_sensor_odometry':
                return False,'WAIT_WORLD_ANCHORED_SENSOR_MAPPING'
            if self.observed_floor is None:return False,'WAIT_OBSERVED_FLOOR'
        positive=('max_vx','max_wz','max_accel','max_angular_accel','goal_tolerance_m','settle_sec',
                  'max_pose_age_sec','input_timeout_sec','max_wall_age_sec','goal_timeout_sec',
                  'stall_timeout_sec','obstacle_timeout_sec','robot_radius_m')
        if any(not math.isfinite(self.p(k)) or self.p(k)<=0 for k in positive):
            return False,'INVALID_NONPOSITIVE_NAVIGATION_PARAMETER'
        if (not .1<=self.p('robot_radius_m')<=2 or not 0<=self.p('safety_margin_m')<=1
                or not 0<self.p('max_vx')<=.15 or not 0<self.p('max_wz')<=.35
                or not self.p('goal_tolerance_m')<=.3 or not self.p('settle_sec')>=.5
                or not -3.<=self.p('lidar_floor_z_m')<=-.2
                or not 0<self.p('obstacle_min_height_m')<self.p('obstacle_max_height_m')<=3.
                or self.p('cloud_point_budget')<1000 or self.p('min_scan_hits')<5):
            return False,'INVALID_OR_EXCESSIVE_MOTION_PARAMETERS'
        if self.map_error is not None:return False,self.map_error
        now=self.now()
        if not self.localization.get('ready') or not self.fresh(self.localization.get('stamp_sim_sec'),self.localization_wall,now):
            return False,'WAIT_FRESH_LOCALIZATION'
        if not self.odometry_status.get('ready') or not self.fresh(self.odometry_status.get('stamp_sim_sec'),self.odom_status_wall,now):
            return False,'WAIT_FRESH_ODOMETRY_STATUS'
        epoch=self.odometry_status.get('epoch')
        if not isinstance(epoch,str) or not epoch or self.localization.get('epoch')!=epoch:
            return False,'LOCALIZATION_ODOMETRY_EPOCH_MISMATCH'
        if self.recovery.request:
            if not self.recovery.eligible(self.odometry_status):
                return False,'RESPAWN_WAIT_NEW_ODOMETRY_EPOCH'
            self.recovery.consume(epoch)
        odom=self.inputs.get('odom')
        if odom is None or not self.fresh(stamp_seconds(odom[0].header.stamp),odom[1],now):
            return False,'WAIT_FRESH_ODOMETRY'
        for key in ('scan','cloud'):
            obs=self.observations.get(key)
            if obs is None or not self.fresh(obs[1],obs[2],now):return False,'WAIT_FRESH_'+key.upper()+'_TF'
            if obs[3]:return False,obs[3]
        try:self.pose()
        except (TransformException,ValueError):return False,'WAIT_FRESH_MAP_TO_PELVIS_TF'
        return True,'READY_FOR_GOAL'

    def publish_status(self,override=None):
        ready,health=self.readiness()
        payload=dict(id=self.goal_id,state=self.state,reason=self.reason,ready=bool(ready),
                     health=health,epoch=self.odometry_status.get('epoch'),
                     map_id=self.localization.get('map_id'),stamp_sim_sec=self.now())
        if override is not None:
            # Replayed terminal results keep their identity/verdict, while
            # readiness, epoch and observation age describe the current robot.
            payload.update({k:override[k] for k in ('id','state','reason','goal_xy') if k in override})
        if self.goal is not None and (override is None or
                ('goal_xy' not in override and override.get('id',self.goal_id)==self.goal_id)):
            payload['goal_xy']=self.goal.tolist()
        if self.last_pose is not None:
            payload['pose_xy']=self.last_pose[:2].tolist()
            if self.goal is not None:
                payload['distance_m']=float(np.linalg.norm(self.goal-self.last_pose[:2]))
        payload['observed_floor_z_m']=self.observed_floor
        msg=String();msg.data=json.dumps(payload,allow_nan=False);self.status_pub.publish(msg)
        return payload

    def reject(self,goal_id,reason):
        payload=self.publish_status(dict(id=goal_id,state='FAILED',reason=reason))
        if goal_id is not None:self.remember(goal_id,payload)

    def remember(self,goal_id,payload):
        self.completed[goal_id]=payload
        while len(self.completed)>30:self.completed.popitem(last=False)

    def on_goal(self,msg):
        try:goal=json.loads(msg.data)
        except (ValueError,TypeError):self.reject(None,'INVALID_GOAL_JSON');return
        self.accept_goal(goal)

    def on_manual_goal(self,msg):
        self.accept_goal(dict(id='rviz-'+uuid.uuid4().hex,frame=msg.header.frame_id,
                              x=msg.pose.position.x,y=msg.pose.position.y))

    def accept_goal(self,goal):
        if not isinstance(goal,dict):self.reject(None,'GOAL_MUST_BE_OBJECT');return
        ident=goal.get('id')
        if not isinstance(ident,str) or not ident.strip() or len(ident)>128:
            self.reject(None,'GOAL_ID_REQUIRED');return
        if ident in self.completed:
            self.publish_status(self.completed[ident]);return
        if self.goal_id==ident:
            self.publish_status();return
        if self.state=='ACTIVE':self.reject(ident,'BUSY_WITH_ANOTHER_GOAL');return
        try:
            if goal.get('frame')!='map':raise ValueError('GOAL_FRAME_MUST_BE_MAP')
            if any(type(goal.get(key)) not in (int,float) for key in ('x','y')):
                raise ValueError('GOAL_XY_MUST_BE_NUMBERS')
            target=np.array([goal['x'],goal['y']],dtype=float)
            if not np.all(np.isfinite(target)):raise ValueError('NONFINITE_GOAL')
            self.process_observations();ready,reason=self.readiness()
            if not ready:raise ValueError(reason)
            if 'epoch' in goal and goal['epoch']!=self.odometry_status.get('epoch'):
                raise ValueError('GOAL_ODOMETRY_EPOCH_MISMATCH')
            if 'map_id' in goal and goal['map_id']!=self.localization.get('map_id'):
                raise ValueError('GOAL_MAP_ID_MISMATCH')
            current=self.pose()
            planner=plan_observed_path if self.p('online_mapping') else plan_path
            try:
                path,free=planner(self.occupancy,self.geom,current[:2],target,
                                    self.p('robot_radius_m')+self.p('safety_margin_m'))
            except ValueError:
                if not self.p('online_mapping'):raise
                path,free=[],None  # hold still while initial map evidence accumulates
        except (ValueError,TypeError,KeyError,TransformException) as exc:
            self.reject(ident,str(exc));return
        self.goal_id,self.goal=ident,target
        self.goal_epoch=self.odometry_status.get('epoch')
        self.path,self.free,self.path_index=path,free,0
        self.state,self.reason='ACTIVE','FOLLOWING_PATH'
        self.started=self.last_progress=self.now();self.best_distance=float(np.linalg.norm(target-current[:2]))
        self.blocked_since=None;self.last_tick=self.now();self.last_command=(0.,0.)
        self.waypoint_best=None
        self.arrival=ArrivalGate(self.p('goal_tolerance_m'),self.p('settle_sec'))
        self.publish_path();self.publish_status()

    def publish_path(self):
        display=Path();display.header.frame_id='map';display.header.stamp=self.get_clock().now().to_msg()
        for xy in self.path:
            p=PoseStamped();p.header=display.header
            p.pose.position.x=float(xy[0]);p.pose.position.y=float(xy[1]);p.pose.orientation.w=1.
            display.poses.append(p)
        self.path_pub.publish(display)

    def on_cancel(self,msg):
        try:payload=json.loads(msg.data)
        except (ValueError,TypeError):return
        if not isinstance(payload,dict):return
        ident=payload.get('id')
        if not isinstance(ident,str) or not ident.strip() or len(ident)>128:return
        if ident in self.completed:self.publish_status(self.completed[ident]);return
        if self.state=='ACTIVE' and ident in (self.goal_id,'*'):self.finish('CANCELED','CANCELED_BY_CLIENT')
        elif ident==self.goal_id:self.publish_status()
        elif ident!='*':
            # Cancellation may arrive before the corresponding goal over DDS.
            # Retain a terminal record so a delayed goal cannot start motion.
            self.remember(ident,self.publish_status(dict(
                id=ident,state='CANCELED',reason='CANCELED_BEFORE_ACCEPTANCE')))

    def command(self,vx,wz):
        msg=Twist();msg.linear.x=float(vx);msg.angular.z=float(wz);self.cmd_pub.publish(msg)
        self.last_command=(float(vx),float(wz))

    def finish(self,state,reason):
        if self.state!='ACTIVE':return
        self.command(0.,0.)
        self.state,self.reason=state,reason
        self.remember(self.goal_id,self.publish_status())

    def control(self):
        self.process_observations()
        if self.state!='ACTIVE':return
        now=self.now()
        if now < self.started:self.finish('FAILED','SIMULATION_TIME_RESET');return
        ready,reason=self.readiness()
        if not ready:
            self.command(0.,0.);self.reason='HOLD: '+reason
            if self.health_lost_wall is None:self.health_lost_wall=time.monotonic()
            if time.monotonic()-self.health_lost_wall>self.p('max_wall_age_sec'):
                self.finish('FAILED',reason)
            return
        self.health_lost_wall=None
        if self.odometry_status.get('epoch')!=self.goal_epoch:
            self.finish('FAILED','ODOMETRY_RESET_REQUIRES_RELOCALIZATION');return
        if now-self.started>self.p('goal_timeout_sec'):
            self.finish('FAILED','GOAL_TIMEOUT');return
        try:pose=self.pose()
        except (TransformException,ValueError):self.finish('FAILED','POSE_LOST');return
        self.last_pose=pose
        if self.p('online_mapping') and (self.map_dirty or not self.path):
            try:
                self.path,self.free=plan_observed_path(self.occupancy,self.geom,pose[:2],self.goal,
                    self.p('robot_radius_m')+self.p('safety_margin_m'))
                self.path_index=0;self.waypoint_best=None;self.map_dirty=False
                self.publish_path()
            except ValueError as exc:
                self.command(0.,0.);self.reason='WAIT_OBSERVED_PATH: '+str(exc)
                if now-self.last_progress>self.p('stall_timeout_sec'):
                    self.finish('FAILED','NO_OBSERVED_PATH: '+str(exc))
                return
        distance=float(np.linalg.norm(self.goal-pose[:2]))
        odom=self.inputs['odom'][0];v=odom.twist.twist.linear;w=odom.twist.twist.angular
        if distance<=self.p('goal_tolerance_m'):
            self.command(0.,0.);self.reason='SETTLING_AT_GOAL'
            if self.arrival.update(now,distance,math.hypot(v.x,v.y),w.z):self.finish('SUCCEEDED','POSITION_AND_STOP_CONFIRMED')
            return
        self.arrival.update(now,distance,math.hypot(v.x,v.y),w.z)
        if distance < self.best_distance-.03:
            self.best_distance,self.last_progress=distance,now
        # Distance-to-goal can temporarily increase on a detour; waypoint progress
        # also refreshes the watchdog, below.
        while self.path_index<len(self.path)-1 and np.linalg.norm(self.path[self.path_index]-pose[:2])<.15:
            self.path_index+=1;self.last_progress=now;self.waypoint_best=None
        waypoint=self.path[self.path_index]
        waypoint_distance=float(np.linalg.norm(waypoint-pose[:2]))
        if (self.p('online_mapping') and self.path_index==len(self.path)-1
                and waypoint_distance<.2):
            self.map_dirty=True
        if self.waypoint_best is None or waypoint_distance<self.waypoint_best-.03:
            self.waypoint_best,self.last_progress=waypoint_distance,now
        if now-self.last_progress>self.p('stall_timeout_sec'):
            self.finish('FAILED','NO_PROGRESS');return
        if not segment_free(self.free,self.geom,pose[:2],waypoint):
            self.finish('FAILED','POSE_LEFT_PLANNED_FREE_CORRIDOR');return
        desired=waypoint_command(pose,waypoint,self.goal,self.p('max_vx'),self.p('max_wz'),
                                 min_vx=min(.11,self.p('max_vx')))
        dt=max(0.,now-self.last_tick);self.last_tick=now
        command=limit_command(self.last_command,desired,dt,self.p('max_accel'),self.p('max_angular_accel'))
        radius=self.p('robot_radius_m')+self.p('safety_margin_m')+.075
        collision=any(swept_collision(self.observations[key][0],pose,*command,radius,
                                      deceleration=self.p('max_accel')) for key in ('scan','cloud'))
        if collision:
            self.command(0.,0.);self.reason='LIVE_OBSTACLE_STOP'
            if self.blocked_since is None:self.blocked_since=now
            if now-self.blocked_since>self.p('obstacle_timeout_sec'):self.finish('FAILED','OBSTACLE_DID_NOT_CLEAR')
            return
        self.blocked_since=None;self.reason='FOLLOWING_PATH';self.command(*command)

    def wall_tick(self):
        if self.state=='ACTIVE':
            now=time.monotonic()
            if any(key not in self.inputs or now-self.inputs[key][1]>self.p('max_wall_age_sec')
                   for key in ('scan','cloud','odom')):
                self.finish('FAILED','STALE_INPUT_OR_SIMULATION_PAUSED')
        self.publish_status()

    def close(self):
        if self.state=='ACTIVE':self.finish('CANCELED','NODE_SHUTDOWN')


def main(args=None):
    from haic_m1_sensors.runtime import run_node
    run_node(Navigator, args)
