"""Publish world->map ONLY for reviewed, map-bound landmark calibration."""
import json
import math
from pathlib import Path
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.clock import Clock,ClockType
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster
from .alignment_core import validate_alignment


class AlignmentNode(Node):
    def __init__(self):
        super().__init__('haic_m1_alignment')
        self.declare_parameter('data_dir','custom/m1_data')
        self.publisher=self.create_publisher(String,'/haic/alignment_status',10)
        self.broadcaster=StaticTransformBroadcaster(self)
        self.pose=None;self.map_id=None;self.ready=False;self.reason='WAIT_CALIBRATION'
        self.clock_wall=Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(1.,self.tick,clock=self.clock_wall)

    def tick(self):
        try:
            directory=Path(self.get_parameter('data_dir').value)
            manifest=json.loads((directory/'map_manifest.json').read_text())
            data=json.loads((directory/'world_alignment.json').read_text())
            pose=validate_alignment(data,manifest['map_id'])
            # A published static transform cannot be safely revoked. A different
            # live map/calibration requires restarting this node and consumers.
            if self.pose is not None and (pose!=self.pose or manifest['map_id']!=self.map_id):
                raise ValueError('Calibration changed: restart bringup before navigation')
            if self.pose is None:
                t=TransformStamped();t.header.stamp=self.get_clock().now().to_msg()
                t.header.frame_id='world';t.child_frame_id='map'
                t.transform.translation.x=float(pose[0]);t.transform.translation.y=float(pose[1])
                t.transform.rotation.z=math.sin(pose[2]/2);t.transform.rotation.w=math.cos(pose[2]/2)
                self.broadcaster.sendTransform(t);self.pose=pose;self.map_id=manifest['map_id']
            self.ready=True;self.reason='ALIGNED'
        except (OSError,ValueError,KeyError,TypeError) as exc:
            self.ready=False;self.reason=str(exc)
        msg=String();msg.data=json.dumps(dict(ready=self.ready,state=self.reason,
            map_id=self.map_id,stamp_sim_sec=self.get_clock().now().nanoseconds*1e-9))
        self.publisher.publish(msg)
        if not self.ready:self.get_logger().warning(self.reason,throttle_duration_sec=10.)


def main(args=None):
    from haic_m1_sensors.runtime import run_node
    run_node(AlignmentNode, args)
