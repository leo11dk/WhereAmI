"""Numerical/input-format checks; runnable without ROS via unittest."""

import math
import struct
from types import SimpleNamespace as NS
import unittest

import numpy as np

from haic_m1_sensors.core import (
    ScanConfig, TiltEstimator, level_rotation, make_scan, matrix_quaternion,
    quaternion_matrix, read_xyz, rotation_vector_matrix,
)


def cloud_fixture(rows, bigendian=False, padding=0, point_step=16):
    order = '>' if bigendian else '<'
    raw = b''
    for row in rows:
        for point in row:
            raw += struct.pack(order+'fffH',*point,3) + b'\0'*(point_step-14)
        raw += b'\xff'*padding
    return NS(width=len(rows[0]),height=len(rows),point_step=point_step,
              row_step=len(rows[0])*point_step+padding,is_bigendian=bigendian,
              fields=[NS(name=n,offset=4*i,datatype=7,count=1)
                      for i,n in enumerate(('x','y','z'))]
                     +[NS(name='ring',offset=12,datatype=4,count=1)],data=raw)


class PointCloudTests(unittest.TestCase):
    def test_unaligned_14_byte_xyz_ring_records(self):
        cloud=cloud_fixture([[(1,2,3),(4,5,6)]],point_step=14)
        np.testing.assert_array_equal(read_xyz(cloud),[[1,2,3],[4,5,6]])

    def test_padded_rows_and_big_endian(self):
        cloud=cloud_fixture([[(1,2,3),(4,5,6)],[(7,8,9),(-1,-2,-3)]],True,7)
        np.testing.assert_array_equal(read_xyz(cloud),[[1,2,3],[4,5,6],[7,8,9],[-1,-2,-3]])

    def test_mixed_float64_offsets(self):
        cloud=NS(width=1,height=1,point_step=28,row_step=28,is_bigendian=False,
                 fields=[NS(name='x',offset=4,datatype=8,count=1),
                         NS(name='y',offset=12,datatype=8,count=1),
                         NS(name='z',offset=20,datatype=8,count=1)],
                 data=b'xxxx'+struct.pack('<ddd',1.25,-2.5,3.75))
        np.testing.assert_array_equal(read_xyz(cloud),[[1.25,-2.5,3.75]])

    def test_truncated_buffer_rejected(self):
        cloud=cloud_fixture([[(1,2,3)]]);cloud.data=cloud.data[:-1]
        with self.assertRaises(ValueError):read_xyz(cloud)

    def test_invalid_field_rejected(self):
        cloud=cloud_fixture([[(1,2,3)]]);cloud.fields[0].offset=100
        with self.assertRaises(ValueError):read_xyz(cloud)

    def test_empty_cloud(self):
        cloud=NS(width=0,height=1)
        self.assertEqual(read_xyz(cloud).shape,(0,3))


class GeometryTests(unittest.TestCase):
    def test_level_rotation_and_tf_direction(self):
        sensor_to_world=rotation_vector_matrix([0.17,0.23,-0.11])
        up_sensor=sensor_to_world.T@np.array([0,0,1.])
        level=level_rotation(up_sensor)
        np.testing.assert_allclose(level@up_sensor,[0,0,1],atol=1e-12)
        np.testing.assert_allclose(level@level.T,np.eye(3),atol=1e-12)
        q=matrix_quaternion(level.T)  # TF child pose in parent
        sample=np.array([2.,-1.,0.5])
        np.testing.assert_allclose(quaternion_matrix(q).T@sample,level@sample,atol=1e-12)

    def test_quaternion_roundtrip_including_pi(self):
        rng=np.random.default_rng(114)
        vectors=[[0,0,0],[math.pi,0,0],[0,math.pi,0],[0,0,math.pi]]
        vectors.extend(rng.normal(size=(100,3)))
        for v in vectors:
            matrix=rotation_vector_matrix(v)
            np.testing.assert_allclose(quaternion_matrix(matrix_quaternion(matrix)),matrix,atol=1e-12)

    def test_vertical_forward_rejected(self):
        with self.assertRaises(ValueError):level_rotation([1,0,0])

    def test_synthetic_tilted_wall_observations(self):
        # Four horizontal-wall returns with known ranges; robot pitches 13deg.
        physical=np.array([[3,0,0],[0,4,0],[-5,0,0],[0,-6,0]],dtype=float)
        sensor_to_level=rotation_vector_matrix([0,math.radians(13),0])
        observed=physical@sensor_to_level
        up_sensor=sensor_to_level.T@np.array([0,0,1.])
        corrected=observed@level_rotation(up_sensor).T
        np.testing.assert_allclose(corrected,physical,atol=1e-12)
        scan,_=make_scan(corrected,ScanConfig())
        np.testing.assert_allclose(scan[[360,540,0,180]],[3,4,5,6],atol=1e-6)


class ScanTests(unittest.TestCase):
    def test_cardinal_axes_nearest_return_and_pi_wrap(self):
        p=np.array([[2,0,0],[1,0,0],[0,3,0],[-4,0,0],[-4,-0.,0],[0,-5,0]])
        ranges,used=make_scan(p,ScanConfig())
        self.assertEqual(used,6)
        np.testing.assert_array_equal(ranges[[360,540,0,180]],[1,3,4,5])
        self.assertEqual(np.isfinite(ranges).sum(),4)

    def test_height_range_and_nan_filters(self):
        p=np.array([[3,0,0],[2,0,-1.3],[2,0,1.],[0.1,0,0],
                    [40,0,0],[float('nan'),1,0],[8,0,0]])
        ranges,used=make_scan(p,ScanConfig())
        self.assertEqual(used,2)
        self.assertEqual(ranges[360],3)

    def test_far_wall_is_not_dropped_at_4m(self):
        scan,_=make_scan(np.array([[12.,0,0]]),ScanConfig())
        self.assertEqual(scan[360],12)

    def test_empty_is_unknown_not_fake_obstacle(self):
        scan,used=make_scan(np.empty((0,3)),ScanConfig())
        self.assertEqual(used,0)
        self.assertTrue(np.isposinf(scan).all())

    def test_invalid_config(self):
        for args in ({'bins':0},{'range_max':-1},{'min_height':1,'max_height':-1}):
            with self.assertRaises(ValueError):ScanConfig(**args)


def calibrated_filter(up=(0,0,1)):
    estimator=TiltEstimator(calibration_sec=0.1)
    for i in range(21):estimator.update(i*.01,[0,0,0],9.81*np.asarray(up))
    return estimator


class TiltTests(unittest.TestCase):
    def test_stationary_tilt_initialization(self):
        up=np.array([math.sin(.2),0,math.cos(.2)])
        estimator=calibrated_filter(up)
        np.testing.assert_allclose(estimator.up,up,atol=1e-12)

    def test_motion_prevents_initialization(self):
        estimator=TiltEstimator(calibration_sec=.1)
        for i in range(100):self.assertIsNone(estimator.update(i*.01,[0,0,1],[0,0,9.81]))

    def test_gyro_propagation_sign(self):
        estimator=calibrated_filter()
        # Reject accelerometer correction so this tests actual gyro propagation.
        estimator.update(.21,[0,1,0],[0,0,30])
        expected=rotation_vector_matrix([0,-.01,0])@np.array([0,0,1.])
        np.testing.assert_allclose(estimator.up,expected,atol=1e-12)
        self.assertFalse(estimator.accel_correction_used)

    def test_pure_yaw_does_not_invent_tilt(self):
        estimator=calibrated_filter()
        for i in range(21,100):estimator.update(i*.01,[0,0,1],[0,0,9.81])
        np.testing.assert_allclose(estimator.up,[0,0,1],atol=1e-12)

    def test_time_jump_requires_recalibration(self):
        estimator=calibrated_filter()
        self.assertIsNone(estimator.update(0.,[0,0,0],[0,0,9.81]))
        self.assertIsNone(estimator.up)

    def test_gap_requires_recalibration(self):
        estimator=calibrated_filter()
        self.assertIsNone(estimator.update(2.,[0,0,0],[0,0,9.81]))
        self.assertIsNone(estimator.up)

    def test_duplicate_and_invalid_imu(self):
        estimator=calibrated_filter();old=estimator.up.copy()
        self.assertIsNone(estimator.update(.2,[0,100,0],[9.81,0,0]))
        np.testing.assert_array_equal(old,estimator.up)
        with self.assertRaises(ValueError):estimator.update(.21,[float('nan'),0,0],[0,0,9.81])


if __name__=='__main__':unittest.main()
