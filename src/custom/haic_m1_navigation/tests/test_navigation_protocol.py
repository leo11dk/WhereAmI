"""Exercise the real ROS adapter's message handlers without a ROS installation."""

import ast
from collections import OrderedDict, deque
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from haic_m1_navigation.recovery import RecoveryGate


path = Path(__file__).resolve().parents[1] / 'haic_m1_navigation' / 'navigation_node.py'
tree = ast.parse(path.read_text())
namespace = {'Node': object, 'json': json, 'np': np, 'math': math, 'TransformException': RuntimeError}
definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)


class NavigationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.node = object.__new__(namespace['Navigator'])
        node = self.node
        node.goal_id = None
        node.state = 'IDLE'
        node.completed = OrderedDict()
        node.responses = []
        node.odometry_status = {'epoch': 'session:1'}
        node.localization = {'map_id': 'current-map'}
        node.inputs = {'scan': object()}
        node.observation_queue = {'scan': deque(), 'cloud': deque()}
        node.observations = {'cloud': object()}
        node.processed = {'scan': 1.}
        node.recovery = RecoveryGate()
        node.now = lambda: 10.
        node.process_observations = lambda: None
        node.readiness = lambda: (True, 'READY')
        node.pose = lambda: self.fail('rejected goals must not reach path planning')

        def publish(override=None):
            response = dict(id=node.goal_id, state=node.state)
            response.update(override or {})
            node.responses.append(response)
            return response

        node.publish_status = publish

    def goal(self, **overrides):
        return dict(id='request', frame='map', x=1., y=2., **overrides)

    def test_cancel_before_goal_prevents_late_motion(self):
        self.node.on_cancel(SimpleNamespace(data=json.dumps({'id': 'request'})))
        self.node.accept_goal(self.goal())
        self.assertEqual(self.node.responses[-1]['state'], 'CANCELED')
        self.assertEqual(self.node.state, 'IDLE')

    def test_old_epoch_request_is_rejected(self):
        self.node.accept_goal(self.goal(epoch='session:0'))
        self.assertEqual(self.node.responses[-1]['reason'], 'GOAL_ODOMETRY_EPOCH_MISMATCH')

    def test_other_map_request_is_rejected(self):
        self.node.accept_goal(self.goal(map_id='previous-map'))
        self.assertEqual(self.node.responses[-1]['reason'], 'GOAL_MAP_ID_MISMATCH')

    def test_malformed_goal_does_not_crash_callback(self):
        for invalid in (None, [], {'id': []}, {'id': '   '},
                        {'id': 'request', 'frame': None, 'x': 1, 'y': 2},
                        {'id': 'request', 'frame': 'map', 'x': True, 'y': 2}):
            with self.subTest(goal=invalid):
                self.node.completed.clear()
                self.node.accept_goal(invalid)
                self.assertEqual(self.node.responses[-1]['state'], 'FAILED')

    def test_malformed_cancel_does_not_crash_callback(self):
        for ident in (None, [], {}, 3, ' '):
            self.node.on_cancel(SimpleNamespace(data=json.dumps({'id': ident})))
        self.assertFalse(self.node.responses)

    def test_public_respawn_stops_active_goal_and_discards_previous_observations(self):
        self.node.state = 'ACTIVE'
        stopped = []
        self.node.finish = lambda *args: stopped.append(args)
        self.node.on_respawn(None)
        self.assertEqual(stopped, [('FAILED', 'PUBLIC_RESPAWN_REQUIRES_NEW_GOAL')])
        self.assertEqual(self.node.recovery.request, 'respawn')
        self.assertFalse(self.node.recovery.eligible(self.node.odometry_status))
        self.assertFalse(self.node.inputs)
        self.assertFalse(self.node.observations)
        self.assertFalse(self.node.processed)

    def test_delayed_tf_uses_recent_transformable_scan_instead_of_starving(self):
        def scan(second):
            return SimpleNamespace(header=SimpleNamespace(frame_id='level',
                stamp=SimpleNamespace(sec=second,nanosec=0)), ranges=[2.]*40,
                angle_min=-2.,angle_increment=.1,range_min=.3,range_max=30.)
        node=self.node
        node.observation_queue['scan'].extend([(scan(1),1.),(scan(2),2.)])
        node.processed={};node.observations={}
        node.now=lambda:2.1
        node.fresh=lambda *a:True
        node.p=lambda name:5
        def transform(frame,stamp):
            if stamp.sec==2:raise RuntimeError('future TF not available yet')
            return np.eye(3),np.zeros(3)
        node.transform=transform
        namespace['Navigator'].process_observations(node)
        self.assertEqual(node.processed['scan'],1.)
        self.assertEqual(node.observations['scan'][1],1.)


if __name__ == '__main__':
    unittest.main()
