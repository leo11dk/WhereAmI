import math
from types import SimpleNamespace as NS
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from haic_m1_navigation.geometry import (
    classify_scan_stamp,se2_matrix,matrix_pose,transform_points,scan_points,wrap_angle,pelvis_from_scan_pose,
)
from haic_m1_navigation.icp import register_points


def room_points(seed=28):
    rng=np.random.default_rng(seed)
    y=rng.uniform(-4,5,250);x=rng.uniform(-3,4,250)
    return np.vstack([np.c_[np.full(250,-3),y],np.c_[np.full(250,4),y],
                      np.c_[x,np.full(250,-4)],np.c_[x,np.full(250,5)]])


class GeometryTests(unittest.TestCase):
    def test_epoch_boundary_rejects_delayed_pre_respawn_scan(self):
        self.assertEqual(classify_scan_stamp(9.9,None,10.0,10.2),'before_epoch')
        self.assertEqual(classify_scan_stamp(10.1,10.2,10.0,10.3),'out_of_order')
        self.assertEqual(classify_scan_stamp(10.2,10.2,10.0,10.3),'duplicate')
        self.assertEqual(classify_scan_stamp(10.4,10.2,10.0,10.4),'accept')

    def test_true_clock_reset_and_old_clock_packet(self):
        self.assertEqual(classify_scan_stamp(.05,100.0,99.0,.06),'clock_reset')
        self.assertEqual(classify_scan_stamp(99.9,100.0,99.0,.06),'wait_clock')
        self.assertEqual(classify_scan_stamp(.05,None,.05,.06),'accept')
        self.assertEqual(classify_scan_stamp(0.,None,0.,0.),'accept')

    def test_bad_timestamp_rejected(self):
        self.assertEqual(classify_scan_stamp(float('nan'),None,0.,0.),'invalid')

    def test_transform_direction_and_inverse(self):
        transform=se2_matrix([2,-1,math.pi/2])
        points=np.array([[1,0],[0,1],[0,0.]])
        np.testing.assert_allclose(transform_points(transform,points),[[2,0],[1,-1],[2,-1]],atol=1e-12)
        np.testing.assert_allclose(transform_points(np.linalg.inv(transform),transform_points(transform,points)),points,atol=1e-12)

    def test_roundtrip_and_angle_wrap(self):
        pose=[1.2,-.7,-math.pi+.03]
        np.testing.assert_allclose(matrix_pose(se2_matrix(pose)),pose,atol=1e-12)
        self.assertAlmostEqual(wrap_angle(2*math.pi+.2),.2)

    def test_scan_keeps_correct_bearings(self):
        msg=NS(ranges=[1,float('inf'),2,0.,float('nan')],range_min=.3,range_max=10,
               angle_min=0,angle_increment=math.pi/2)
        np.testing.assert_allclose(scan_points(msg),[[1,0],[-2,0]],atol=1e-12)

    def test_scan_subsampling_is_bounded(self):
        msg=NS(ranges=[2.]*1000,range_min=.3,range_max=10,angle_min=-math.pi,
               angle_increment=2*math.pi/1000)
        self.assertEqual(scan_points(msg,100).shape,(100,2))

    def test_pelvis_offset_and_tilt_are_not_discarded(self):
        # Build a physically consistent full pose with head offset and tilt.
        pelvis=np.eye(4);pelvis[:3,:3]=Rotation.from_euler('xyz',[.12,.23,.7]).as_matrix()
        pelvis[:3,3]=[3,4,0]
        r_scan=Rotation.from_euler('z',.9).as_matrix()
        scan_offset=np.array([.05,-.02,.8])
        extrinsic=np.eye(4)
        extrinsic[:3,:3]=pelvis[:3,:3].T@r_scan
        extrinsic[:3,3]=scan_offset
        scan_xy=(pelvis[:3,3]+pelvis[:3,:3]@scan_offset)[:2]
        recovered=pelvis_from_scan_pose([*scan_xy,.9],extrinsic)
        np.testing.assert_allclose(recovered,pelvis,atol=1e-12)

    def test_invalid_scan_angles(self):
        msg=NS(ranges=[1]*20,range_min=.3,range_max=10,angle_min=0,angle_increment=0)
        with self.assertRaises(ValueError):scan_points(msg)


class RegistrationTests(unittest.TestCase):
    def test_known_source_to_target_transform(self):
        target=room_points()
        truth=se2_matrix([.12,-.08,.035])
        source=transform_points(np.linalg.inv(truth),target)
        result=register_points(source,target)
        self.assertTrue(result.accepted,result.reason)
        np.testing.assert_allclose(result.transform,truth,atol=2e-3)

    def test_noise_and_distant_outliers(self):
        target=room_points();rng=np.random.default_rng(19)
        truth=se2_matrix([.12,-.08,.035])
        source=transform_points(np.linalg.inv(truth),target)+rng.normal(0,.006,target.shape)
        source=np.vstack([source,rng.uniform(20,30,size=(250,2))])
        result=register_points(source,target)
        self.assertTrue(result.accepted,result.reason)
        np.testing.assert_allclose(matrix_pose(result.transform),[.12,-.08,.035],atol=.006)

    def test_nonidentity_initial_guess(self):
        target=room_points()
        truth=se2_matrix([1.4,-.8,.4])
        source=transform_points(np.linalg.inv(truth),target)
        initial=se2_matrix([1.35,-.75,.38])
        result=register_points(source,target,initial=initial)
        self.assertTrue(result.accepted,result.reason)
        np.testing.assert_allclose(result.transform,truth,atol=.003)

    def test_parallel_corridor_rejected_even_at_zero_residual(self):
        y=np.linspace(-5,5,100)
        points=np.vstack([np.c_[np.zeros(100),y],np.c_[np.full(100,3.),y]])
        result=register_points(points,points)
        self.assertFalse(result.accepted)
        self.assertTrue(result.degenerate)
        self.assertEqual(result.reason,'DEGENERATE_GEOMETRY')
        self.assertLess(result.rmse,1e-6)

    def test_no_overlap_rejected(self):
        target=room_points()
        result=register_points(target+100,target)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason,'INSUFFICIENT_CORRESPONDENCES')

    def test_missing_and_nonfinite_points(self):
        result=register_points(np.array([[0,0],[1,1],[np.nan,2]]),np.array([[0,0],[1,1]]))
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason,'INSUFFICIENT_POINTS')

    def test_bad_parameters_rejected(self):
        with self.assertRaises(ValueError):register_points(room_points(),room_points(),trim_fraction=0)

    def test_composed_motion_direction(self):
        a=se2_matrix([1,0,math.pi/2]);b=se2_matrix([1,0,0])
        np.testing.assert_allclose(matrix_pose(a@b),[1,1,math.pi/2],atol=1e-12)


if __name__=='__main__':unittest.main()
