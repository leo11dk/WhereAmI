"""Meaningful map-planning, collision and stopping tests without ROS 2."""
import math
import unittest

import numpy as np

from haic_m1_navigation.planning import (
    ArrivalGate,GridGeometry,astar,inflated_free_grid,limit_command,plan_path,
    segment_free,swept_collision,waypoint_command,
)


class PlanningTests(unittest.TestCase):
    def test_origin_rotation_and_roundtrip(self):
        g=GridGeometry(.1,3.,-2.,math.pi/2)
        np.testing.assert_allclose(g.center((0,0)),[2.95,-1.95],atol=1e-12)
        for c in ((0,0),(10,20),(200,5)):
            self.assertEqual(g.cell(g.center(c)),c)

    def test_unknown_and_map_border_inflation(self):
        grid=np.zeros((30,30),dtype=int);grid[15,15]=-1
        free=inflated_free_grid(grid,.1,.3)
        self.assertFalse(free[15,15]);self.assertFalse(free[15,17])
        self.assertFalse(free[0,10]);self.assertTrue(free[8,8])

    def test_uncertain_probability_is_blocked(self):
        grid=np.zeros((30,30),dtype=int);grid[15,15]=50
        self.assertFalse(inflated_free_grid(grid,.1,0.)[15,15])

    def test_astar_will_not_cut_diagonal_corner(self):
        free=np.ones((3,3),dtype=bool);free[0,1]=free[1,0]=False
        with self.assertRaisesRegex(ValueError,'NO_COLLISION_FREE_PATH'):
            astar(free,(0,0),(1,1))

    def test_path_detours_obstacle_and_every_segment_is_free(self):
        grid=np.zeros((60,60),dtype=int);grid[15:45,28:32]=100
        geom=GridGeometry(.1,0.,0.)
        path,free=plan_path(grid,geom,[1.,3.],[5.,3.],.2)
        self.assertGreaterEqual(len(path),3)
        self.assertTrue(any(y<1.5 or y>4.5 for x,y in path))
        self.assertTrue(all(segment_free(free,geom,a,b) for a,b in zip(path,path[1:])))

    def test_unknown_barrier_blocks_path(self):
        grid=np.zeros((20,20),dtype=int);grid[:,10]=-1
        with self.assertRaises(ValueError):plan_path(grid,GridGeometry(.1,0,0),[.5,1.],[1.5,1.],.1)

    def test_outside_or_occupied_goal_rejected(self):
        grid=np.zeros((30,30),dtype=int);grid[15,15]=100
        for goal in ([1.55,1.55],[8.,8.]):
            with self.assertRaisesRegex(ValueError,'BLOCKED_OR_OUTSIDE'):
                plan_path(grid,GridGeometry(.1,0,0),[.8,.8],goal,.2)

    def test_supercover_diagonal_touches_blocked_side(self):
        free=np.ones((4,4),dtype=bool);free[0,1]=False
        self.assertFalse(segment_free(free,GridGeometry(1.,0,0),[.5,.5],[2.5,2.5]))

    def test_grid_boundary_segment_checks_both_sides_along_entire_length(self):
        geom=GridGeometry(1.,0,0)
        for blocked,start,end in (((3,1),[2.,1.2],[2.,4.8]),
                                   ((1,3),[1.2,2.],[4.8,2.])):
            free=np.ones((6,6),dtype=bool);free[blocked]=False
            self.assertFalse(segment_free(free,geom,start,end))
            self.assertFalse(segment_free(free,geom,end,start))

    def test_negative_going_exact_endpoint_does_not_overshoot(self):
        free=np.ones((60,60),dtype=bool)
        self.assertTrue(segment_free(free,GridGeometry(.1,0,0),[4.95,3.15],[5.,3.]))

    def test_straight_path_is_shortcut_to_two_points(self):
        grid=np.zeros((50,50),dtype=int)
        path,_=plan_path(grid,GridGeometry(.1,0,0),[1.,1.],[4.,4.],.1)
        self.assertEqual(len(path),2)


class ControlTests(unittest.TestCase):
    def test_robot_facing_away_rotates_without_forward_motion(self):
        vx,wz=waypoint_command([0,0,math.pi],[2,0],[2,0])
        self.assertEqual(vx,0);self.assertLessEqual(abs(wz),.35)

    def test_left_and_right_rotation_sign_and_angle_wrap(self):
        self.assertGreater(waypoint_command([0,0,0],[1,1],[1,1])[1],0)
        self.assertLess(waypoint_command([0,0,0],[1,-1],[1,-1])[1],0)
        vx,wz=waypoint_command([0,0,math.pi-.01],[-1.,-.01],[-1.,-.01])
        self.assertGreater(vx,0);self.assertLess(abs(wz),.1)

    def test_speed_reduces_near_goal(self):
        far=waypoint_command([0,0,0],[2,0],[2,0])[0]
        near=waypoint_command([0,0,0],[.25,0],[.25,0])[0]
        self.assertLess(near,far);self.assertLessEqual(far,.15)

    def test_acceleration_and_angular_acceleration_limits(self):
        v,w=limit_command((0.,0.),(.15,.35),.1)
        self.assertAlmostEqual(v,.015);self.assertAlmostEqual(w,.04)
        self.assertEqual(limit_command((.1,.1),(0.,0.),0.),(.1,.1))

    def test_front_capsule_and_turn_disk_obstacles(self):
        self.assertTrue(swept_collision(np.array([[.65,0.]]),[0,0,0],.15,0,.5))
        self.assertFalse(swept_collision(np.array([[2.,0.],[0.,1.]]),[0,0,0],.15,0,.5))
        self.assertTrue(swept_collision(np.array([[0.,.4]]),[0,0,0],0.,.3,.5))
        self.assertFalse(swept_collision(np.array([[-1.,0.]]),[0,0,0],.15,0,.5))

    def test_collision_transforms_with_robot_heading(self):
        self.assertTrue(swept_collision(np.array([[1.,1.65]]),[1,1,math.pi/2],.15,0,.5))
        self.assertFalse(swept_collision(np.array([[1.8,1.]]),[1,1,math.pi/2],.15,0,.5))

    def test_arrival_requires_stopped_and_sustained_inside(self):
        gate=ArrivalGate()
        self.assertFalse(gate.update(0.,.1,.1,0.))
        self.assertFalse(gate.update(1.,.1,0.,0.))
        self.assertFalse(gate.update(1.5,.3,0.,0.))
        self.assertFalse(gate.update(2.,.1,0.,0.))
        self.assertFalse(gate.update(2.9,.1,0.,0.))
        self.assertTrue(gate.update(3.01,.1,0.,0.))

    def test_time_reset_cannot_complete_arrival(self):
        gate=ArrivalGate();gate.update(100.,.1,0.,0.)
        self.assertFalse(gate.update(1.,.1,0.,0.))


if __name__=='__main__':unittest.main()
