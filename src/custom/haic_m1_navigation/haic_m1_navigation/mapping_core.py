"""Bounded 2D occupancy mapping and portable, atomic map snapshot persistence.

Finite LaserScan returns provide free-space evidence along the ray and occupied
evidence at its endpoint. Nonfinite/invalid beams remain unobserved: the HAIC
height-filtered scan does not promise that an infinite range means clear floor.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np


def timestamp_in_epoch(stamp,epoch_start):
    """Fail closed for missing/invalid epoch boundaries; allow float roundoff."""
    return (isinstance(stamp,(int,float)) and isinstance(epoch_start,(int,float))
            and math.isfinite(stamp) and math.isfinite(epoch_start)
            and stamp+1e-9>=epoch_start)


@dataclass(frozen=True)
class GridConfig:
    resolution: float = 0.10
    origin_x: float = -30.0
    origin_y: float = -30.0
    width: int = 600
    height: int = 600

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.resolution,self.origin_x,self.origin_y)):
            raise ValueError('map geometry must be finite')
        if self.resolution <= 0 or not isinstance(self.width,int) or not isinstance(self.height,int):
            raise ValueError('positive resolution and integer dimensions required')
        if self.width < 2 or self.height < 2 or self.width*self.height > 2_000_000:
            raise ValueError('map must contain 4..2000000 cells')


def bresenham(x0,y0,x1,y1):
    """Integer grid cells traversed by a line, including both endpoints."""
    dx,dy=abs(x1-x0),-abs(y1-y0)
    sx,sy=(1 if x0<x1 else -1),(1 if y0<y1 else -1)
    error=dx+dy
    while True:
        yield x0,y0
        if x0==x1 and y0==y1:
            break
        twice=2*error
        if twice>=dy:
            error+=dy;x0+=sx
        if twice<=dx:
            error+=dx;y0+=sy


class OccupancyMap:
    def __init__(self,config=None):
        self.config=config or GridConfig()
        self.log_odds=np.zeros((self.config.height,self.config.width),dtype=np.float32)
        self.observed=np.zeros_like(self.log_odds,dtype=bool)
        self.scans=0
        self.hit_rays=0

    def cell(self,point):
        x,y=np.asarray(point,dtype=float)
        c=self.config
        return (int(math.floor((x-c.origin_x)/c.resolution)),
                int(math.floor((y-c.origin_y)/c.resolution)))

    def inside(self,x,y):
        return 0<=x<self.config.width and 0<=y<self.config.height

    def _clipped_endpoint(self,start,end):
        """Clip a ray whose origin is inside; return world endpoint and hit flag."""
        c=self.config
        lower=np.array([c.origin_x,c.origin_y])
        upper=lower+np.array([c.width,c.height])*c.resolution
        hit=bool(np.all(end>=lower) and np.all(end<upper))
        if hit:return end,True
        delta=end-start
        fraction=1.0
        epsilon=c.resolution*1e-6
        for axis in range(2):
            if delta[axis]>0:
                fraction=min(fraction,(upper[axis]-epsilon-start[axis])/delta[axis])
            elif delta[axis]<0:
                fraction=min(fraction,(lower[axis]+epsilon-start[axis])/delta[axis])
        return start+max(0.0,fraction)*delta,False

    def update(self,origin,endpoints):
        origin=np.asarray(origin,dtype=float)
        endpoints=np.asarray(endpoints,dtype=float)
        if origin.shape!=(2,) or not np.all(np.isfinite(origin)):
            raise ValueError('finite XY scan origin required')
        if endpoints.ndim!=2 or endpoints.shape[1]!=2:
            raise ValueError('endpoints must have shape Nx2')
        x0,y0=self.cell(origin)
        if not self.inside(x0,y0):
            raise ValueError('scan origin outside map; enlarge bounds and remap')
        endpoints=endpoints[np.all(np.isfinite(endpoints),axis=1)]
        if not len(endpoints):return 0
        free_cells=set();hit_cells=set();hits=0
        width=self.config.width
        for end in endpoints:
            clipped,hit=self._clipped_endpoint(origin,end)
            x1,y1=self.cell(clipped)
            cells=list(bresenham(x0,y0,x1,y1))
            for x,y in cells[:-1] if hit else cells:
                if self.inside(x,y):free_cells.add(y*width+x)
            if hit and self.inside(x1,y1):
                hit_cells.add(y1*width+x1);hits+=1
        # An occupied endpoint wins if another sampled ray crossed its cell.
        free_cells.difference_update(hit_cells)
        log=self.log_odds.ravel();observed=self.observed.ravel()
        if free_cells:
            idx=np.fromiter(free_cells,dtype=np.intp)
            log[idx]=np.maximum(-3.0,log[idx]-0.40);observed[idx]=True
        if hit_cells:
            idx=np.fromiter(hit_cells,dtype=np.intp)
            log[idx]=np.minimum(3.0,log[idx]+0.85);observed[idx]=True
        self.scans+=1;self.hit_rays+=hits
        return hits

    def occupancy(self):
        result=np.full(self.log_odds.shape,-1,dtype=np.int8)
        probability=1/(1+np.exp(-self.log_odds[self.observed]))
        result[self.observed]=np.rint(100*probability).astype(np.int8)
        return result

    def occupied_points(self,threshold=65,max_points=20000):
        y,x=np.nonzero(self.occupancy()>=threshold)
        if len(x)>max_points:
            choose=np.linspace(0,len(x)-1,max_points,dtype=int);x,y=x[choose],y[choose]
        c=self.config
        return np.column_stack((c.origin_x+(x+.5)*c.resolution,
                                c.origin_y+(y+.5)*c.resolution))


def _atomic_bytes(path,payload):
    tmp=path.with_name(path.name+'.tmp_'+uuid.uuid4().hex)
    try:
        with tmp.open('wb') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def save_map(grid,directory,metadata=None,min_scans=5):
    """Commit a snapshot through one atomic manifest after all versioned files.

    The manifest is the authoritative pointer. Existing snapshots are retained.
    PGM rows are top-to-bottom; ROS occupancy rows run from lowest Y upwards.
    """
    if grid.scans<min_scans or len(grid.occupied_points())<20:
        raise ValueError(f'not enough map evidence: need >={min_scans} scans and >=20 occupied cells')
    directory=Path(directory).expanduser().resolve();directory.mkdir(parents=True,exist_ok=True)
    snapshot='map_'+uuid.uuid4().hex[:12]
    npz=directory/(snapshot+'.npz');pgm=directory/(snapshot+'.pgm');yaml=directory/(snapshot+'.yaml')
    c=grid.config;extra=dict(metadata or {})
    meta=dict(schema='haic-m1-map-v1',map_id=snapshot,resolution=c.resolution,
              origin=[c.origin_x,c.origin_y],width=c.width,height=c.height,
              scans=grid.scans,hit_rays=grid.hit_rays,metadata=extra)
    # NPZ is written without pickle; stored JSON is a scalar unicode array.
    temporary=npz.with_name(npz.name+'.tmp_'+uuid.uuid4().hex)
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,log_odds=grid.log_odds,observed=grid.observed,
                                metadata_json=json.dumps(meta,allow_nan=False))
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,npz)
    finally:
        if temporary.exists():temporary.unlink()
    occ=grid.occupancy();pixels=np.full(occ.shape,205,dtype=np.uint8)
    pixels[(occ>=0)&(occ<=35)]=254;pixels[occ>=65]=0
    image=np.flipud(pixels).tobytes()
    _atomic_bytes(pgm,f'P5\n{c.width} {c.height}\n255\n'.encode()+image)
    text=(f'image: {pgm.name}\nmode: trinary\nresolution: {c.resolution:.12g}\n'
          f'origin: [{c.origin_x:.12g}, {c.origin_y:.12g}, 0.0]\n'
          'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
    _atomic_bytes(yaml,text.encode())
    manifest={**meta,'schema':'haic-m1-map-manifest-v1','npz':npz.name,'yaml':yaml.name,
              'pgm':pgm.name,'npz_sha256':hashlib.sha256(npz.read_bytes()).hexdigest()}
    _atomic_bytes(directory/'map_manifest.json',json.dumps(manifest,indent=2,allow_nan=False).encode())
    return manifest


def load_map(directory):
    directory=Path(directory).expanduser().resolve()
    manifest=json.loads((directory/'map_manifest.json').read_text())
    if manifest.get('schema')!='haic-m1-map-manifest-v1':
        raise ValueError('unsupported map manifest')
    filename=manifest.get('npz','')
    if not filename or Path(filename).name!=filename:
        raise ValueError('invalid map filename in manifest')
    path=directory/filename
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest.get('npz_sha256'):
        raise ValueError('map checksum mismatch; preserve files and remap/resave')
    with np.load(path,allow_pickle=False) as saved:
        meta=json.loads(str(saved['metadata_json']))
        if meta.get('schema')!='haic-m1-map-v1':raise ValueError('unsupported map format')
        if not meta.get('map_id') or meta['map_id']!=manifest.get('map_id'):
            raise ValueError('map identity mismatch in manifest')
        origin=meta['origin']
        config=GridConfig(resolution=float(meta['resolution']),origin_x=float(origin[0]),
                          origin_y=float(origin[1]),width=int(meta['width']),height=int(meta['height']))
        grid=OccupancyMap(config)
        log=np.asarray(saved['log_odds'],dtype=np.float32)
        observed=np.asarray(saved['observed'],dtype=bool)
        if log.shape!=grid.log_odds.shape or observed.shape!=log.shape or not np.all(np.isfinite(log)):
            raise ValueError('invalid occupancy data shape or values')
        grid.log_odds=log.copy();grid.observed=observed.copy()
        grid.scans=int(meta['scans']);grid.hit_rays=int(meta['hit_rays'])
    return grid,meta
