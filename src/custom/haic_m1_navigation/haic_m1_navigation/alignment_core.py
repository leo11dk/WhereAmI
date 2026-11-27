"""World/map calibration from operator-identified FIXED landmarks, not GT topics."""
import json
import math
import os
from pathlib import Path
import numpy as np


def fit_alignment(map_points, world_points, max_rms_m=.15):
    a=np.asarray(map_points,dtype=float);b=np.asarray(world_points,dtype=float)
    if a.ndim!=2 or a.shape[1]!=2 or a.shape!=b.shape or len(a)<2:
        raise ValueError('At least two corresponding map/world XY landmarks are required')
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('Landmarks must be finite')
    if min(np.linalg.norm(a-a[0],axis=1).max(),np.linalg.norm(b-b[0],axis=1).max())<2.:
        raise ValueError('Choose landmarks at least 2 metres apart')
    ac=a-a.mean(axis=0);bc=b-b.mean(axis=0)
    u,_,vt=np.linalg.svd(ac.T@bc)
    r=vt.T@u.T
    if np.linalg.det(r)<0:vt[-1]*=-1;r=vt.T@u.T
    t=b.mean(axis=0)-r@a.mean(axis=0)
    errors=np.linalg.norm(a@r.T+t-b,axis=1)
    rms=float(np.sqrt(np.mean(errors**2)))
    if rms>max_rms_m:raise ValueError(f'Landmark residual {rms:.3f}m exceeds {max_rms_m}m')
    return dict(world_from_map=[float(t[0]),float(t[1]),math.atan2(r[1,0],r[0,0])],
                rms_m=rms,landmark_count=len(a))


def validate_alignment(data,map_id):
    if not isinstance(data,dict) or data.get('verified') is not True:
        raise ValueError('World/map landmarks have not been operator-verified')
    if data.get('map_id')!=map_id:raise ValueError('Calibration belongs to a different map; recalibrate')
    pose=data.get('world_from_map')
    if not isinstance(pose,list) or len(pose)!=3 or not all(
            isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in pose):
        raise ValueError('Invalid world_from_map [x,y,yaw_rad]')
    if not math.isfinite(data.get('rms_m',math.inf)) or data['rms_m']>.15:
        raise ValueError('Calibration residual too high')
    return pose


def calibrate_file(data_dir,landmarks_file,confirm=False):
    directory=Path(data_dir)
    manifest=json.loads((directory/'map_manifest.json').read_text())
    pairs=json.loads(Path(landmarks_file).read_text())
    result=fit_alignment(pairs['map_points'],pairs['world_points'])
    if not confirm:raise ValueError('Review actual fixed landmark identities, then use --confirm-landmarks')
    result.update(map_id=manifest['map_id'],verified=True,units='m,rad',
                  note='Operator identified fixed landmarks; fit residual alone is not accuracy proof')
    target=directory/'world_alignment.json'
    tmp=target.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2))
    os.replace(tmp,target)
    return result
