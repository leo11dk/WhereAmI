import math
import unittest
import numpy as np
from haic_m1_navigation.recovery import RecoveryGate,spawn_in_map
from haic_m1_navigation.geometry import matrix_pose,se2_matrix


class RecoveryTests(unittest.TestCase):
    def status(self,**kw):
        return dict(ready=True,epoch='session:1',epoch_start_sim_sec=40.,
                    epoch_reason='RESPAWN_NEW_ODOM_EPOCH',**kw)

    def test_fixed_world_spawn_converts_with_inverse_alignment(self):
        alignment=dict(verified=True,map_id='map-a',rms_m=.01,
                       world_from_map=[3.,-4.,math.pi/2])
        got=spawn_in_map(alignment,'map-a')
        np.testing.assert_allclose(got,[15.,-13.,-math.pi/2],atol=1e-10)
        np.testing.assert_allclose(matrix_pose(se2_matrix(alignment['world_from_map'])@
            se2_matrix(got)),[16.,11.,0.],atol=1e-10)

    def test_unverified_or_other_map_is_rejected(self):
        data=dict(verified=True,map_id='a',rms_m=.01,world_from_map=[0.,0.,0.])
        for change in (dict(verified=False),dict(map_id='b'),dict(rms_m=.3),
                       dict(world_from_map=[0.,0.,float('nan')])):
            with self.assertRaises(ValueError):spawn_in_map({**data,**change},'a')

    def test_no_event_means_no_automatic_reset(self):
        gate=RecoveryGate()
        self.assertFalse(gate.eligible(self.status()))

    def test_empty_before_or_after_new_odometry_status(self):
        # Persistent epoch_reason supports either inter-topic delivery order.
        for event_time in (39.98,40.02):
            gate=RecoveryGate();gate.respawn(event_time)
            self.assertTrue(gate.eligible(self.status()))

    def test_old_epoch_or_unrelated_tracking_reset_cannot_use_spawn(self):
        gate=RecoveryGate();gate.respawn(40.)
        for update in (dict(epoch_start_sim_sec=30.),dict(epoch_reason='TRACKING_GAP_NEW_ODOM_EPOCH'),
                       dict(ready=False),dict(epoch=''),dict(epoch_start_sim_sec=float('nan'))):
            self.assertFalse(gate.eligible({**self.status(),**update}))

    def test_consumed_event_does_not_reseed_same_epoch(self):
        gate=RecoveryGate();gate.respawn(40.)
        self.assertTrue(gate.eligible(self.status()))
        gate.consume('session:1')
        self.assertFalse(gate.eligible(self.status()))
        gate.respawn(40.1)
        self.assertFalse(gate.eligible(self.status()))

    def test_manual_seed_cancels_automatic_seed(self):
        gate=RecoveryGate(True);gate.respawn(40.);gate.cancel()
        self.assertFalse(gate.eligible(self.status()))

    def test_startup_requires_explicit_opt_in_and_first_epoch(self):
        data={**self.status(), 'epoch_reason':'WAIT_SCAN'}
        self.assertFalse(RecoveryGate().eligible(data))
        gate=RecoveryGate(True)
        self.assertTrue(gate.eligible(data))
        self.assertFalse(gate.eligible(self.status()))

if __name__=='__main__':unittest.main()
