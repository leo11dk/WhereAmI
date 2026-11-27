import math,unittest
import numpy as np
from haic_m1_navigation.alignment_core import fit_alignment,validate_alignment

class AlignmentTests(unittest.TestCase):
    def test_known_rotation_translation(self):
        a=np.array([[0,0],[3,0],[0,4.]])
        yaw=.7;r=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
        b=a@r.T+[16,11]
        d=fit_alignment(a,b)
        np.testing.assert_allclose(d['world_from_map'],[16,11,yaw],atol=1e-10)
    def test_not_enough_geometry(self):
        with self.assertRaises(ValueError):fit_alignment([[0,0],[.1,0]],[[0,0],[.1,0]])
    def test_bad_correspondence_scale(self):
        with self.assertRaises(ValueError):fit_alignment([[0,0],[3,0]],[[0,0],[7,0]])
    def test_calibration_cannot_follow_remap(self):
        d=dict(world_from_map=[0,0,0],verified=True,map_id='old',rms_m=0)
        with self.assertRaises(ValueError):validate_alignment(d,'new')
        d['verified']=False
        with self.assertRaises(ValueError):validate_alignment(d,'old')

if __name__=='__main__':unittest.main()
