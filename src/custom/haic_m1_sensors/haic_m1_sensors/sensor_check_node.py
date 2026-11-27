"""Observe sensor freshness, rates and frame connectivity without driving G1."""

from __future__ import annotations

from collections import deque
import json
import time

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState, LaserScan, PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


class SensorCheck(Node):
    def __init__(self):
        super().__init__('haic_sensor_check')
        for key, value in dict(report_period_sec=2.0, stale_wall_sec=3.0,
                               stale_sim_sec=0.5, rate_window_wall_sec=4.0).items():
            self.declare_parameter(key,value)
        self.history = {}
        self.last = {}
        self.previous_sim = None
        self.preprocess = {}
        self.preprocess_wall = None
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer,self)
        topics = {
            '/lidar/points': PointCloud2, '/lidar/imu': Imu, '/imu': Imu,
            '/camera/color/image_raw': Image, '/camera/camera_info': CameraInfo,
            '/joint_states': JointState, '/scan': LaserScan,
        }
        for topic, msg_type in topics.items():
            self.history[topic] = deque(maxlen=3000)
            self.create_subscription(msg_type, topic,
                lambda msg,t=topic: self.observe(t,msg), qos_profile_sensor_data)
        self.create_subscription(String,'/haic/preprocess_status',self.on_preprocess,10)
        self.status_pub = self.create_publisher(String,'/haic/sensor_status',10)
        self.wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(self.get_parameter('report_period_sec').value,
                          self.report,clock=self.wall_clock)

    def observe(self, topic, msg):
        wall = time.monotonic()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        history = self.history[topic]
        if history and stamp < history[-1][1]:
            history.clear()
        history.append((wall,stamp))
        frame = msg.header.frame_id
        self.last[topic] = dict(stamp=stamp,wall=wall,frame=frame)

    def on_preprocess(self,msg):
        try:
            payload = json.loads(msg.data)
        except (ValueError,TypeError):
            return
        if isinstance(payload,dict):
            self.preprocess = payload
            self.preprocess_wall = time.monotonic()

    def report(self):
        now = time.monotonic()
        sim = self.get_clock().now().nanoseconds*1e-9
        if self.previous_sim is not None and sim < self.previous_sim:
            for history in self.history.values():
                history.clear()
            self.last.clear()
            self.preprocess = {}
            self.preprocess_wall = None
        self.previous_sim = sim
        stale_wall = self.get_parameter('stale_wall_sec').value
        stale_sim = self.get_parameter('stale_sim_sec').value
        window = self.get_parameter('rate_window_wall_sec').value
        report = {}
        for topic,history in self.history.items():
            while history and now-history[0][0] > window:
                history.popleft()
            last = self.last.get(topic)
            wall_age = None if last is None else now-last['wall']
            sim_age = None if last is None else sim-last['stamp']
            fresh = (last is not None and wall_age < stale_wall
                     and -0.1 <= sim_age < stale_sim)
            hz_sim = hz_wall = 0.0
            if len(history) > 1:
                dw = history[-1][0]-history[0][0]
                ds = history[-1][1]-history[0][1]
                hz_wall = (len(history)-1)/dw if dw>0 else 0.0
                hz_sim = (len(history)-1)/ds if ds>0 else 0.0
            report[topic] = dict(fresh=bool(fresh),
                frame=None if last is None else last['frame'],
                hz_sim=round(hz_sim,1),hz_wall=round(hz_wall,1),
                wall_age_sec=None if wall_age is None else round(wall_age,3),
                sim_age_sec=None if sim_age is None else round(sim_age,3))
        tf_ready = {}
        for child in ('lidar_link','imu_link','camera_color_optical_frame'):
            try:
                self.buffer.lookup_transform('pelvis',child,Time())
                tf_ready['pelvis_to_'+child] = True
            except TransformException:
                tf_ready['pelvis_to_'+child] = False
        mapping_topics = ('/lidar/points','/lidar/imu','/joint_states')
        inputs_ready = all(report[t]['fresh'] for t in mapping_topics) and all(tf_ready.values())
        status_fresh = (self.preprocess_wall is not None
                        and now-self.preprocess_wall < stale_wall)
        preprocessing_ready = (inputs_ready and report['/scan']['fresh'] and status_fresh
                               and self.preprocess.get('ready') is True)
        payload = dict(stage=1, inputs_ready=bool(inputs_ready),
            preprocessing_ready=bool(preprocessing_ready),
            # This stage does not implement odometry, a map, or localization.
            navigation_implemented=False, topics=report, tf=tf_ready,
            preprocessing_state=self.preprocess.get('state','WAIT_PREPROCESS_STATUS'))
        msg = String()
        msg.data = json.dumps(payload,allow_nan=False)
        self.status_pub.publish(msg)
        rates = ' '.join(t+'='+str(report[t]['hz_sim'])+'Hz'
                         +('' if report[t]['fresh'] else '(STALE)') for t in mapping_topics+('/scan',))
        state = 'PREPROCESS_READY' if preprocessing_ready else payload['preprocessing_state']
        if not inputs_ready and state == 'PREPROCESS_READY':
            state = 'WAIT_SENSOR_INPUTS_OR_TF'
        self.get_logger().info('[Stage1] '+state+' | '+rates+' | map/odom: next stage')


def main(args=None):
    from .runtime import run_node
    run_node(SensorCheck, args)
