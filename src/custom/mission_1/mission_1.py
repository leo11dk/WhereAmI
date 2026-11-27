#!/usr/bin/env python3
"""Mission 1 adapter: validated world goal -> guarded navigation -> referee.

The runner owns ROS initialization/spinning. This module never reads scenario
files, uses ground-truth odometry, teleports the robot or calls the judge.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import threading
import time
import uuid

SRC_ROOT=Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:sys.path.insert(0,str(SRC_ROOT))

from mission_api import FAIL, Mission, MissionResult, run_standalone  # noqa:E402


def finite_goal(value):
    if (not isinstance(value,(list,tuple)) or len(value) not in (2,3)
            or any(type(v) not in (float,int) or not math.isfinite(v) for v in value)):
        raise ValueError('mission 1 needs finite world goal [x,y]')
    x,y=map(float,value[:2])
    return x,y


def world_to_map_xy(x,y,transform):
    """Transform a world XY point using map<-world, not its inverse."""
    t=transform.transform
    q=t.rotation
    # Alignment is planar SE(2), not a GT robot pose.
    norm=math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)
    if not math.isfinite(norm) or norm<1e-10:
        raise ValueError('invalid map<-world rotation')
    qx,qy,qz,qw=q.x/norm,q.y/norm,q.z/norm,q.w/norm
    if abs(qx)+abs(qy)>1e-5 or not all(math.isfinite(v) for v in (t.translation.x,t.translation.y)):
        raise ValueError('world/map alignment must be finite and planar')
    yaw=math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
    c,s=math.cos(yaw),math.sin(yaw)
    return c*x-s*y+t.translation.x,s*x+c*y+t.translation.y


def status_fresh(payload,received_wall,now_wall,now_sim,wall_limit=3.0,sim_limit=1.5):
    if not isinstance(payload,dict) or payload.get('ready') is not True or received_wall is None:
        return False
    stamp=payload.get('stamp_sim_sec')
    return (type(stamp) in (int,float) and math.isfinite(stamp)
            and 0 <= now_wall-received_wall <= wall_limit
            and -0.1 <= now_sim-stamp <= sim_limit)


def valid_epoch(epoch):
    # Session UUID plus reset counter: a new odometry process must not reuse an
    # old integer epoch after restart, even if its local counter returns to 0.
    return isinstance(epoch,str) and 0 < len(epoch) <= 128 and epoch==epoch.strip()


class Mission1(Mission):
    MISSION=1

    def __init__(self):
        super().__init__()
        from rclpy.parameter import Parameter
        from std_msgs.msg import Empty, String
        self.set_parameters([Parameter('use_sim_time',Parameter.Type.BOOL,True)])
        self._String=String
        self._state_lock=threading.Condition()
        self._states={}
        self._respawns=0
        self.create_subscription(Empty,'/g1/respawned',self._on_respawn,10,
                                 callback_group=self._api_group)
        self._goal_pub=self.create_publisher(String,'/haic/navigation_goal',10)
        self._cancel_pub=self.create_publisher(String,'/haic/navigation_cancel',10)
        for name in ('localization','alignment','navigation'):
            self.create_subscription(String,'/haic/'+name+'_status',
                lambda msg,n=name:self._on_navigation_state(n,msg),10,
                callback_group=self._api_group)
        for name,default in {
            'ready_timeout_sec':45.0,'mission_timeout_sec':300.0,
            'status_stale_wall_sec':3.0,'localization_stale_sim_sec':1.5,
            'arrival_tolerance_m':0.25,'arrival_settle_sec':0.3,
        }.items():self.declare_parameter(name,default)
        for name in ('ready_timeout_sec','mission_timeout_sec','status_stale_wall_sec',
                     'localization_stale_sim_sec','arrival_tolerance_m','arrival_settle_sec'):
            value=float(self.get_parameter(name).value)
            if not math.isfinite(value) or value<=0:
                raise ValueError(name+' must be finite and positive')
        if float(self.get_parameter('arrival_tolerance_m').value)>0.35:
            raise ValueError('arrival_tolerance_m must preserve margin below the 0.5 m judge radius')
        self._tf()  # subscribe early; do not wait for TF in a callback

    def _on_navigation_state(self,name,msg):
        try:payload=json.loads(msg.data)
        except (ValueError,TypeError):return
        if not isinstance(payload,dict):return
        with self._state_lock:
            self._states[name]=(payload,time.monotonic())
            self._state_lock.notify_all()

    def _state(self,name):
        with self._state_lock:return self._states.get(name,({},None))

    def _on_respawn(self,_msg):
        with self._state_lock:
            self._respawns+=1
            self._state_lock.notify_all()

    def _respawn_count(self):
        with self._state_lock:return self._respawns

    def _fresh(self,name,sim_limit=None):
        payload,receipt=self._state(name)
        return status_fresh(payload,receipt,time.monotonic(),self.get_clock().now().nanoseconds*1e-9,
            wall_limit=float(self.get_parameter('status_stale_wall_sec').value),
            sim_limit=sim_limit or float(self.get_parameter('localization_stale_sim_sec').value))

    def _wait_tick(self):
        with self._state_lock:self._state_lock.wait(timeout=0.05)

    def _cancel(self,goal_id):
        if not goal_id:return
        msg=self._String(data=json.dumps({'id':goal_id}))
        # Bounded retries while callbacks keep running in the framework thread.
        for _ in range(3):
            if not self.context.ok():return
            self._cancel_pub.publish(msg)
            time.sleep(0.05)

    def _wait_ready(self,gx,gy,deadline,previous_epoch=None):
        from rclpy.time import Time
        while time.monotonic()<deadline:
            localization=self._state('localization')[0]
            epoch=localization.get('epoch')
            map_id=localization.get('map_id')
            if (self._fresh('localization') and self._fresh('alignment',sim_limit=3.0)
                    and self._fresh('navigation') and self._goal_pub.get_subscription_count()>0
                    and self._state('navigation')[0].get('state')!='ACTIVE'
                    and valid_epoch(epoch) and epoch!=previous_epoch
                    and self._state('navigation')[0].get('epoch')==epoch
                    and isinstance(map_id,str) and bool(map_id)
                    and self._state('alignment')[0].get('map_id')==map_id
                    and self._state('navigation')[0].get('map_id')==map_id):
                try:
                    transform=self._tf().lookup_transform('map','world',Time())
                    return world_to_map_xy(gx,gy,transform),epoch
                except Exception:
                    pass  # bounded below; no pose/config fallback
            self._wait_tick()
        details={n:self._state(n)[0].get('health',self._state(n)[0].get(
                    'reason',self._state(n)[0].get('state','missing')))
                 for n in ('localization','alignment','navigation')}
        raise TimeoutError('navigation not ready: '+json.dumps(details,ensure_ascii=False))

    def _drive(self,goal_id,goal_map,epoch,respawn_count,deadline):
        map_id=self._state('localization')[0].get('map_id')
        request=self._String(data=json.dumps({'id':goal_id,'frame':'map',
            'x':goal_map[0],'y':goal_map[1],'epoch':epoch,'map_id':map_id}))
        sent_at=time.monotonic()
        last_publish=-math.inf;acked=False;settled_since=None
        while time.monotonic()<deadline:
            if self._respawn_count()!=respawn_count:
                raise RuntimeError('public respawn event requires a new navigation goal')
            if not self._fresh('localization') or not self._fresh('alignment',sim_limit=3.0):
                raise RuntimeError('localization or verified world/map alignment lost')
            if self._state('localization')[0].get('epoch')!=epoch:
                raise RuntimeError('odometry epoch changed')
            if (self._state('localization')[0].get('map_id')!=map_id
                    or self._state('alignment')[0].get('map_id')!=map_id):
                raise RuntimeError('world/map calibration no longer matches the localized map')
            nav,received=self._state('navigation')
            if nav.get('id')==goal_id:
                acked=True
                if received is None or time.monotonic()-received>3.0:
                    raise TimeoutError('navigation status became stale')
                state=nav.get('state')
                if state in ('FAILED','CANCELED'):
                    raise RuntimeError('navigation '+state+': '+str(nav.get('reason','')))
                if state=='SUCCEEDED':
                    pose=self.pose(timeout=0.05)
                    if (pose is None or math.hypot(pose[0]-goal_map[0],pose[1]-goal_map[1])>
                            float(self.get_parameter('arrival_tolerance_m').value)):
                        settled_since=None
                    else:
                        if settled_since is None:settled_since=time.monotonic()
                        if time.monotonic()-settled_since>=float(self.get_parameter('arrival_settle_sec').value):
                            return
            elif acked:
                raise RuntimeError('navigation ownership changed to a different goal')
            if not acked and time.monotonic()-last_publish>=1.0:
                self._goal_pub.publish(request);last_publish=time.monotonic()
            if not acked and time.monotonic()>sent_at+10.0:
                raise TimeoutError('navigation did not acknowledge the goal within 10 seconds')
            self._wait_tick()
        raise TimeoutError('mission 1 navigation exceeded mission_timeout_sec')

    def run(self) -> MissionResult:
        goal_id=None
        try:
            self.begin_mission(1,timeout=20.0)  # Nov26 API returns None
            value=self.params.get('goal')
            if value is None:value=self.m1_goal  # practice only; hidden in web mode
            gx,gy=finite_goal(value)
            deadline=time.monotonic()+float(self.get_parameter('mission_timeout_sec').value)
            previous_epoch=None;recoveries=0
            while time.monotonic()<deadline:
                observed=self._respawn_count()
                goal_map,epoch=self._wait_ready(gx,gy,min(deadline,time.monotonic()+
                    float(self.get_parameter('ready_timeout_sec').value)),previous_epoch)
                # A second event during relocalization restarts the readiness wait.
                if self._respawn_count()!=observed:continue
                goal_id=uuid.uuid4().hex
                self.log(f'[mission 1] world goal ({gx:.3f},{gy:.3f}), map goal {goal_map}; request {goal_id}')
                try:
                    self._drive(goal_id,goal_map,epoch,observed,deadline)
                    return self.result(goal_world=[gx,gy],goal_map=list(goal_map),navigation_id=goal_id)
                except (RuntimeError,TimeoutError):
                    self._cancel(goal_id);self.stop()
                    # Public event and own status messages can arrive in either order.
                    grace=min(deadline,time.monotonic()+1.0)
                    while self._respawn_count()==observed and time.monotonic()<grace:
                        self._wait_tick()
                    if self._respawn_count()==observed or recoveries>=2:raise
                    recoveries+=1;previous_epoch=epoch
                    self.log('[mission 1] public respawn: waiting for sensor localization, then replanning')
            raise TimeoutError('mission 1 navigation exceeded mission_timeout_sec')
        except Exception as exc:
            self.log('[mission 1] '+type(exc).__name__+': '+str(exc))
            return MissionResult(1,FAIL,type(exc).__name__+': '+str(exc))
        finally:
            self._cancel(goal_id)
            self.stop()


if __name__=='__main__':raise SystemExit(run_standalone(Mission1))
