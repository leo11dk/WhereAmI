"""World-anchored sensor mapping, initialized only at the documented spawn.

The initial (16,11,0) is a rulebook reference, not a simulator pose readout.
After initialization all motion comes from LiDAR odometry and sensor TF.
An unrelated odometry reset never implies a return to the starting position.
"""
import json
import math
import time
import uuid

import numpy as np
from geometry_msgs.msg import TransformStamped
from rclpy.time import Time
from std_msgs.msg import Empty, String
from tf2_ros import StaticTransformBroadcaster, TransformException

from .geometry import matrix_pose, se2_matrix, scan_points, transform_points
from .mapping_core import timestamp_in_epoch
from .mapping_node import MappingNode, _map_message, _stamp, _transform_2d
from .recovery import RecoveryGate


def anchor_at_spawn(odom_from_pelvis):
    return se2_matrix([16., 11., 0.]) @ np.linalg.inv(odom_from_pelvis)


class OnlineMapping(MappingNode):
    def __init__(self):
        super().__init__()
        self.p['max_scan_points'] = 720
        self.anchor = None
        self.recovery = RecoveryGate(auto_start=True)
        self.map_id = 'online_' + uuid.uuid4().hex
        self.localization_pub = self.create_publisher(String, '/haic/localization_status', 10)
        self.alignment_pub = self.create_publisher(String, '/haic/alignment_status', 10)
        self.static = StaticTransformBroadcaster(self)
        identity = TransformStamped()
        identity.header.frame_id = 'world'
        identity.child_frame_id = 'map'
        identity.transform.rotation.w = 1.
        self.static.sendTransform(identity)
        self.create_subscription(Empty, '/g1/respawned', self.on_respawn, 10)
        self.reason = 'WAIT_INITIAL_SPAWN_SENSOR_ODOMETRY'

    def on_odometry(self, msg):
        try:
            status = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        if not isinstance(status, dict):
            return
        epoch = status.get('epoch')
        if epoch != self.epoch:
            self.anchor = None
            self.pending.clear()
            self.accepted_stamp = self.accepted_wall = None
            self.epoch = epoch
        self.odometry, self.odometry_wall = status, time.monotonic()

    def on_respawn(self, _msg):
        self.recovery.respawn(self.get_clock().now().nanoseconds * 1e-9)
        self.anchor = None
        self.pending.clear()
        self.accepted_stamp = self.accepted_wall = None
        self.reason = 'RESPAWN_WAIT_NEW_ODOMETRY'

    def process(self):
        if not self.odometry_ready():
            self.reason = 'WAIT_FRESH_ACCEPTED_ODOMETRY'
            return
        if self.anchor is None:
            if not self.recovery.eligible(self.odometry):
                self.reason = 'ODOMETRY_RESET_REQUIRES_KNOWN_SPAWN'
                return
            try:
                tf = self.buffer.lookup_transform('odom', 'pelvis', Time())
                if not timestamp_in_epoch(_stamp(tf.header.stamp),
                                          self.odometry.get('epoch_start_sim_sec')):
                    return
                self.anchor = anchor_at_spawn(_transform_2d(tf))
            except (TransformException, ValueError):
                return
            self.recovery.consume(self.epoch)
        if not self.pending:
            return
        # Scan matching already integrates every scan. Mapping can use the most
        # recent one instead of building a backlog while ray casting a grid.
        scan, received = self.pending[-1]
        stamp = _stamp(scan.header.stamp)
        now = self.get_clock().now().nanoseconds * 1e-9
        if (time.monotonic() - received > self.p['max_wall_age_sec']
                or not timestamp_in_epoch(stamp, self.odometry.get('epoch_start_sim_sec'))
                or not -.1 <= now - stamp < self.p['max_sensor_age_sec']
                or (self.accepted_stamp is not None and stamp <= self.accepted_stamp)):
            self.pending.clear()
            return
        try:
            tf = self.buffer.lookup_transform('odom', scan.header.frame_id,
                                              Time.from_msg(scan.header.stamp))
            transform = self.anchor @ _transform_2d(tf)
            points = scan_points(scan, self.p['max_scan_points'])
            endpoints = transform_points(transform, points)
        except (TransformException, ValueError, TypeError):
            return
        self.pending.clear()
        self.grid.update(transform[:2, 2], endpoints)
        self.accepted_stamp, self.accepted_wall = stamp, time.monotonic()
        out = TransformStamped()
        out.header.stamp = scan.header.stamp
        out.header.frame_id, out.child_frame_id = 'map', 'odom'
        x, y, yaw = matrix_pose(self.anchor)
        out.transform.translation.x, out.transform.translation.y = float(x), float(y)
        out.transform.rotation.z, out.transform.rotation.w = math.sin(yaw/2), math.cos(yaw/2)
        self.broadcaster.sendTransform(out)
        if time.monotonic() - self.last_map_wall >= .5:
            self.map_pub.publish(_map_message(self.grid, 'map', scan.header.stamp))
            self.last_map_wall = time.monotonic()
        self.reason = 'MAPPING_ACTIVE'
        self.publish_status()

    def publish_status(self):
        ready = (self.anchor is not None and self.accepted_stamp is not None
                 and self.accepted_wall is not None and self.grid.scans >= 5
                 and time.monotonic()-self.accepted_wall < self.p['max_wall_age_sec']
                 and self.odometry_ready())
        status = dict(ready=bool(ready), state=self.reason, map_id=self.map_id,
                      epoch=self.epoch, stamp_sim_sec=self.accepted_stamp,
                      source='documented_spawn_sensor_odometry', scans=self.grid.scans)
        message = String(data=json.dumps(status))
        self.status_pub.publish(message)
        self.localization_pub.publish(message)
        self.alignment_pub.publish(message)


def main(args=None):
    from haic_m1_sensors.runtime import run_node
    run_node(OnlineMapping, args)
