"""Keep simulator internals and referee-only channels out of robot controllers."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class MissionRuleTests(unittest.TestCase):
    def sources(self):
        for directory in ('mission_1','haic_m1_sensors/haic_m1_sensors',
                          'haic_m1_navigation/haic_m1_navigation'):
            for path in (ROOT/directory).glob('*.py'):
                yield path,ast.parse(path.read_text())

    def test_controllers_do_not_import_simulator_or_referee_internals(self):
        forbidden={'mujoco','ros2_bridge','referee','environment','scenarios'}
        for path,tree in self.sources():
            for node in ast.walk(tree):
                modules=[]
                if isinstance(node,ast.Import):modules=[alias.name for alias in node.names]
                if isinstance(node,ast.ImportFrom) and node.module:modules=[node.module]
                for module in modules:
                    self.assertNotIn(module.split('.')[0],forbidden,str(path))

    def test_no_private_judge_subscription_or_direct_judge_commands(self):
        forbidden={'/g1/mission_status','/g1/mission_end','/g1/randomize',
                   '/g1/respawn','/g1/mission_begin'}
        for path,tree in self.sources():
            for node in ast.walk(tree):
                if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                        and node.func.attr in ('create_subscription','create_publisher')
                        and len(node.args)>1 and isinstance(node.args[1],ast.Constant)):
                    self.assertNotIn(node.args[1].value,forbidden,str(path))
