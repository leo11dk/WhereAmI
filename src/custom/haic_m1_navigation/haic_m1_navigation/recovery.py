"""Recovery uses the public respawn event and the documented fixed spawn only."""
import math
import numpy as np
from .alignment_core import validate_alignment
from .geometry import matrix_pose, se2_matrix


def spawn_in_map(alignment,map_id):
    """Documented world pelvis (16,11), yaw0 -> current saved map frame."""
    world_from_map=validate_alignment(alignment,map_id)
    return matrix_pose(np.linalg.inv(se2_matrix(world_from_map))@se2_matrix([16.,11.,0.]))


class RecoveryGate:
    """Correlate asynchronous Empty and odometry reset without a private topic."""
    def __init__(self,auto_start=False):
        self.request='startup' if auto_start else None
        self.event_sim=None;self.used_epoch=None

    def respawn(self,stamp):
        self.request='respawn';self.event_sim=float(stamp)

    def cancel(self):
        self.request=None

    def eligible(self,status):
        if not self.request or not status.get('ready'):return False
        epoch=status.get('epoch');start=status.get('epoch_start_sim_sec')
        if not isinstance(epoch,str) or not epoch or epoch==self.used_epoch:return False
        if not isinstance(start,(int,float)) or not math.isfinite(start):return False
        if self.request=='respawn':
            return (status.get('epoch_reason')=='RESPAWN_NEW_ODOM_EPOCH'
                    and self.event_sim is not None and abs(start-self.event_sim)<=1.)
        # A tracking-loss epoch does not imply the robot returned to spawn.
        return status.get('epoch_reason')=='WAIT_SCAN'

    def consume(self,epoch):
        self.used_epoch=epoch;self.request=None
