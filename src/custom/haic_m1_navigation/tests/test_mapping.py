"""ROS-independent occupancy, saved-map and seeded localization checks."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from haic_m1_navigation.geometry import se2_matrix,transform_points,matrix_pose
from haic_m1_navigation.icp import register_points
from haic_m1_navigation.mapping_core import (
    GridConfig,OccupancyMap,bresenham,save_map,load_map,timestamp_in_epoch,
)


class GridTests(unittest.TestCase):
    def grid(self):return OccupancyMap(GridConfig(1.,0.,0.,10,10))

    def test_hit_free_unknown_semantics(self):
        grid=self.grid();grid.update([.5,.5],np.array([[4.5,.5]]))
        occ=grid.occupancy()
        self.assertGreaterEqual(occ[0,4],65)
        self.assertLess(occ[0,2],50)
        self.assertEqual(occ[1,4],-1)
        self.assertEqual(occ[0,5],-1)

    def test_row_is_y_column_is_x(self):
        grid=self.grid();grid.update([.5,.5],np.array([[.5,4.5]]))
        self.assertGreaterEqual(grid.occupancy()[4,0],65)
        self.assertEqual(grid.occupancy()[0,4],-1)

    def test_unknown_beams_do_not_clear(self):
        grid=self.grid()
        self.assertEqual(grid.update([.5,.5],np.array([[np.inf,0],[np.nan,1]])),0)
        self.assertTrue((grid.occupancy()==-1).all())
        self.assertEqual(grid.scans,0)

    def test_clipped_ray_is_free_not_fake_boundary_wall(self):
        grid=self.grid();grid.update([.5,.5],np.array([[50.,.5]]))
        self.assertTrue((grid.occupancy()[0]<50).all())
        self.assertFalse((grid.occupancy()>=65).any())

    def test_origins_outside_rejected(self):
        grid=self.grid()
        with self.assertRaises(ValueError):grid.update([-.1,.5],np.array([[1.,1.]]))

    def test_hit_wins_within_same_scan(self):
        grid=self.grid();grid.update([.5,.5],np.array([[2.5,.5],[5.5,.5]]))
        self.assertGreaterEqual(grid.occupancy()[0,2],65)

    def test_negative_map_origin_and_cell_centers(self):
        grid=OccupancyMap(GridConfig(.5,-2.,-3.,10,10))
        grid.update([-1.75,-2.75],np.array([[-.75,-2.75]]))
        self.assertEqual(grid.cell([-.75,-2.75]),(2,0))
        np.testing.assert_allclose(grid.occupied_points(),[[-.75,-2.75]])

    def test_ray_traversal_both_directions(self):
        forward=list(bresenham(0,0,4,2));reverse=list(bresenham(4,2,0,0))
        self.assertEqual(forward[0],(0,0));self.assertEqual(forward[-1],(4,2))
        self.assertEqual(reverse[0],(4,2));self.assertEqual(reverse[-1],(0,0))
        self.assertTrue(all(abs(a[0]-b[0])<=1 and abs(a[1]-b[1])<=1
                            for a,b in zip(forward,forward[1:])))

    def test_log_odds_saturates(self):
        grid=self.grid()
        for _ in range(100):grid.update([.5,.5],np.array([[4.5,.5]]))
        self.assertTrue(np.max(grid.log_odds)<=3.)
        self.assertTrue(np.min(grid.log_odds)>=-3.)

    def test_invalid_bounds_rejected(self):
        for kw in ({'resolution':0},{'width':5000,'height':5000},{'origin_x':float('nan')}):
            with self.assertRaises(ValueError):GridConfig(**kw)


class EpochTests(unittest.TestCase):
    def test_delayed_old_scans_rejected(self):
        self.assertFalse(timestamp_in_epoch(12.4,12.5))
        self.assertTrue(timestamp_in_epoch(12.5,12.5))
        self.assertTrue(timestamp_in_epoch(12.6,12.5))

    def test_missing_or_nonfinite_boundaries_rejected(self):
        for stamp,start in ((None,0.),(1.,None),(float('nan'),0.),
                            (1.,float('inf')),('1',0.),(1.,'0')):
            self.assertFalse(timestamp_in_epoch(stamp,start))

    def test_boundary_tolerance_and_reset_origin(self):
        self.assertTrue(timestamp_in_epoch(5.-1e-10,5.))
        self.assertFalse(timestamp_in_epoch(5.-1e-7,5.))
        self.assertTrue(timestamp_in_epoch(0.,0.))


def mapped_room():
    grid=OccupancyMap(GridConfig(.1,-6.,-6.,120,120))
    x=np.linspace(-4.,4.,161);y=np.linspace(-3.,3.,121)
    points=np.concatenate((np.column_stack((x,np.full_like(x,-3.))),
        np.column_stack((x,np.full_like(x,3.))),np.column_stack((np.full_like(y,-4.),y)),
        np.column_stack((np.full_like(y,4.),y))))
    for _ in range(5):grid.update([0.,0.],points)
    return grid,points


class PersistenceTests(unittest.TestCase):
    def test_roundtrip_exact_and_portable(self):
        grid,_=mapped_room()
        with tempfile.TemporaryDirectory() as td:
            manifest=save_map(grid,td,{'odometry_epoch':'epoch-A'})
            restored,meta=load_map(td)
            np.testing.assert_array_equal(restored.log_odds,grid.log_odds)
            np.testing.assert_array_equal(restored.observed,grid.observed)
            self.assertEqual(restored.scans,grid.scans)
            self.assertEqual(meta['map_id'],manifest['map_id'])
            self.assertEqual(meta['metadata']['odometry_epoch'],'epoch-A')
            self.assertIn('image: '+manifest['pgm'],(Path(td)/manifest['yaml']).read_text())

    def test_snapshot_pointer_changes_identity(self):
        grid,_=mapped_room()
        with tempfile.TemporaryDirectory() as td:
            first=save_map(grid,td);second=save_map(grid,td)
            self.assertNotEqual(first['map_id'],second['map_id'])
            self.assertEqual(load_map(td)[1]['map_id'],second['map_id'])
            self.assertTrue((Path(td)/first['npz']).exists())

    def test_checksum_rejects_corruption(self):
        grid,_=mapped_room()
        with tempfile.TemporaryDirectory() as td:
            manifest=save_map(grid,td)
            path=Path(td)/manifest['npz'];path.write_bytes(path.read_bytes()+b'corruption')
            with self.assertRaises(ValueError):load_map(td)

    def test_manifest_cannot_swap_map_identity(self):
        grid,_=mapped_room()
        with tempfile.TemporaryDirectory() as td:
            save_map(grid,td);path=Path(td)/'map_manifest.json'
            value=json.loads(path.read_text());value['map_id']='a-different-map'
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):load_map(td)

    def test_insufficient_evidence_does_not_write_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):save_map(OccupancyMap(),td)
            self.assertFalse((Path(td)/'map_manifest.json').exists())

    def test_pgm_flips_vertical_axis(self):
        grid=OccupancyMap(GridConfig(1.,0.,0.,10,10))
        grid.scans=5;grid.observed[0,:]=True;grid.log_odds[0,:]=3.
        grid.observed[1,:]=True;grid.log_odds[1,:]=3.
        grid.observed[9,:]=True;grid.log_odds[9,:]=-3.
        with tempfile.TemporaryDirectory() as td:
            manifest=save_map(grid,td)
            payload=(Path(td)/manifest['pgm']).read_bytes().split(b'\n',3)[3]
            pixels=np.frombuffer(payload,dtype=np.uint8).reshape((10,10))
            self.assertTrue((pixels[0]==254).all())
            self.assertTrue((pixels[-1]==0).all())
            self.assertTrue((pixels[5]==205).all())


class LocalizationGeometryTests(unittest.TestCase):
    def test_seeded_scan_to_occupied_map_registration(self):
        grid,physical=mapped_room()
        truth=se2_matrix([.7,-.4,.12])
        source=transform_points(np.linalg.inv(truth),physical)
        result=register_points(source,grid.occupied_points(),
            initial=se2_matrix([.64,-.35,.10]),max_correspondence_m=.8,max_rmse_m=.20,min_inliers=35)
        self.assertTrue(result.accepted,result.reason)
        np.testing.assert_allclose(matrix_pose(result.transform),matrix_pose(truth),atol=.09)

    def test_seed_map_odom_composition(self):
        map_pelvis=se2_matrix([5.,-2.,.4]);odom_pelvis=se2_matrix([.5,.2,-.1])
        map_odom=map_pelvis@np.linalg.inv(odom_pelvis)
        np.testing.assert_allclose(map_odom@odom_pelvis,map_pelvis,atol=1e-12)


if __name__=='__main__':unittest.main()
