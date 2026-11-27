import ast
from pathlib import Path
import unittest

import numpy as np

from haic_m1_navigation.geometry import se2_matrix, matrix_pose
from haic_m1_navigation.ground import floor_height
from haic_m1_navigation.planning import GridGeometry,plan_observed_path,segment_free,waypoint_command
from haic_m1_navigation.icp import register_scan
from haic_m1_navigation.geometry import transform_points

path = Path(__file__).resolve().parents[1]/'haic_m1_navigation/online_mapping_node.py'
tree = ast.parse(path.read_text())
namespace = dict(np=np, se2_matrix=se2_matrix)
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='anchor_at_spawn')
exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)


class OnlineTests(unittest.TestCase):
    def test_spawn_anchor_uses_inverse_sensor_odometry_pose(self):
        odom = se2_matrix([.12, -.08, .25])
        anchor = namespace['anchor_at_spawn'](odom)
        np.testing.assert_allclose(matrix_pose(anchor@odom), [16, 11, 0], atol=1e-12)

    def test_sensor_motion_is_preserved_after_anchoring(self):
        first = se2_matrix([.1, -.2, .2])
        anchor = namespace['anchor_at_spawn'](first)
        moved = first@se2_matrix([0, 2., 0])
        np.testing.assert_allclose(matrix_pose(anchor@moved), [16, 13, 0], atol=1e-12)

    def test_floor_estimation_ignores_robot_and_high_obstacles(self):
        rng = np.random.default_rng(1)
        floor = np.column_stack((rng.uniform(3,6,500), rng.uniform(-1,1,500),
                                 rng.normal(-1.32,.006,500)))
        objects = np.array([[2,0,-.2], [.2,0,-1.5], [4,0,float('nan')]])
        self.assertAlmostEqual(floor_height(np.vstack([floor,objects])), -1.32, delta=.01)

    def test_missing_floor_does_not_invent_a_height(self):
        with self.assertRaises(ValueError):
            floor_height(np.array([[3,0,-.2]]*100))

    def test_unknown_distant_goal_uses_only_observed_free_path(self):
        grid=np.full((80,100),-1,dtype=int);grid[10:70,10:50]=0
        geom=GridGeometry(.1,0,0)
        path,free=plan_observed_path(grid,geom,[2,4],[8,4],.3)
        self.assertLess(path[-1][0],5.)
        self.assertGreater(path[-1][0],2.)
        self.assertTrue(all(segment_free(free,geom,a,b) for a,b in zip(path,path[1:])))

    def test_arrival_speed_remains_above_provided_gait_phase_threshold(self):
        vx,_=waypoint_command([0,0,0],[.21,0],[.21,0],min_vx=.11)
        self.assertGreater(vx,.1)

    def test_wall_normals_recover_subsample_corridor_translation(self):
        y=np.linspace(-10,10,220)
        target=np.vstack([np.column_stack((np.full_like(y,2),y)),
                          np.column_stack((np.full_like(y,-2),y)),
                          np.column_stack((np.linspace(-2,2,50),np.full(50,10.)))])
        expected=se2_matrix([.005,.018,.004])
        source=transform_points(np.linalg.inv(expected),target)
        result=register_scan(source,target)
        self.assertTrue(result.accepted,result.reason)
        np.testing.assert_allclose(matrix_pose(result.transform),matrix_pose(expected),atol=.002)

    def test_wall_normal_match_rejects_unobservable_parallel_corridor(self):
        y=np.linspace(-10,10,220)
        target=np.vstack([np.column_stack((np.full_like(y,2),y)),
                          np.column_stack((np.full_like(y,-2),y))])
        self.assertFalse(register_scan(target,target).accepted)
