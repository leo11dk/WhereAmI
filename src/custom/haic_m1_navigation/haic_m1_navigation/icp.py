"""Conservative trimmed point-to-point SE2 ICP for the chapter prototypes.

Input source points are expressed in their own frame. Output transform maps
source into target. A low residual alone is insufficient: parallel walls may
have little longitudinal information. We also inspect local normal diversity.
This is a local optimizer, never global localization or loop closure.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial import cKDTree
from .geometry import transform_points, matrix_pose


@dataclass
class ICPResult:
    transform: np.ndarray
    accepted: bool
    reason: str
    rmse: float = math.inf
    inliers: int = 0
    condition: float = math.inf
    degenerate: bool = True
    iterations: int = 0


def _points(value):
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError('expected Nx2 points')
    return p[np.isfinite(p).all(axis=1)]


def _rigid_fit(source, target):
    a, b = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source-a).T @ (target-b))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    out = np.eye(3)
    out[:2,:2], out[:2,2] = r, b-r@a
    return out


def _normal_condition(target, tree, target_indices):
    """Translation information ratio from target-local wall normals."""
    indices = np.unique(target_indices)
    if len(indices) < 3 or len(target) < 6:
        return 0.0, math.inf
    # Bounded work even for large occupancy maps.
    indices = indices[np.linspace(0,len(indices)-1,min(250,len(indices))).astype(int)]
    _, neighbors = tree.query(target[indices], k=min(8,len(target)))
    clouds = target[neighbors]
    centered = clouds-clouds.mean(axis=1,keepdims=True)
    cov = np.einsum('nki,nkj->nij',centered,centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Isolated/random neighborhoods do not provide reliable wall normals.
    reliable = (eigenvalues[:,1] > 1e-8) & (eigenvalues[:,0] <= .35*eigenvalues[:,1])
    if np.count_nonzero(reliable) < 3:
        return 0.0, math.inf
    normals = eigenvectors[reliable,:,0]
    values = np.linalg.eigvalsh(normals.T @ normals)
    ratio = float(values[0]/max(values[-1],1e-12))
    return ratio, float(1/max(ratio,1e-12))


def register_points(source, target, initial=None, max_correspondence_m=.6,
                    trim_fraction=.8, max_iterations=35, min_inliers=20,
                    max_rmse_m=.18, min_normal_eigen_ratio=.01,
                    min_inlier_fraction=.35):
    """Register source→target and return transform plus independent diagnostics.

    Failed result transforms are diagnostic candidates and MUST NOT be applied
    by callers. min_normal_eigen_ratio=0 disables normal rejection for a
    separately verified case; default rejection is intentionally conservative.
    """
    src, dst = _points(source), _points(target)
    t = np.eye(3) if initial is None else np.array(initial,dtype=float,copy=True)
    if t.shape != (3,3) or not np.isfinite(t).all():
        raise ValueError('initial must be finite 3x3')
    if not (0 < trim_fraction <= 1 and 0 <= min_inlier_fraction <= 1
            and max_correspondence_m > 0 and max_rmse_m > 0
            and max_iterations >= 1 and min_inliers >= 3
            and 0 <= min_normal_eigen_ratio <= 1):
        raise ValueError('invalid ICP parameters')
    if min(len(src),len(dst)) < min_inliers:
        return ICPResult(t,False,'INSUFFICIENT_POINTS')
    tree = cKDTree(dst)
    iterations = 0
    for iterations in range(1,max_iterations+1):
        moved = transform_points(t,src)
        distance, nearest = tree.query(moved,k=1)
        selected = np.flatnonzero(distance <= max_correspondence_m)
        if len(selected) < min_inliers:
            return ICPResult(t,False,'INSUFFICIENT_CORRESPONDENCES',iterations=iterations)
        keep = max(min_inliers,int(len(selected)*trim_fraction))
        selected = selected[np.argsort(distance[selected])[:keep]]
        delta = _rigid_fit(moved[selected],dst[nearest[selected]])
        t = delta @ t
        pose = matrix_pose(delta)
        if np.linalg.norm(pose[:2]) < 1e-5 and abs(pose[2]) < 1e-5:
            break
    moved = transform_points(t,src)
    distance, nearest = tree.query(moved,k=1)
    selected = np.flatnonzero(distance <= max_correspondence_m)
    fraction = len(selected)/len(src)
    if len(selected) < min_inliers or fraction < min_inlier_fraction:
        return ICPResult(t,False,'LOW_OVERLAP',inliers=len(selected),iterations=iterations)
    keep = max(min_inliers,int(len(selected)*trim_fraction))
    selected = selected[np.argsort(distance[selected])[:keep]]
    rmse = float(np.sqrt(np.mean(distance[selected]**2)))
    ratio, condition = _normal_condition(dst,tree,nearest[selected])
    degenerate = ratio < max(min_normal_eigen_ratio,1e-10)
    accepted = rmse <= max_rmse_m and (not degenerate or min_normal_eigen_ratio == 0)
    reason = 'MATCH_OK' if accepted else ('DEGENERATE_GEOMETRY' if degenerate else 'HIGH_RESIDUAL')
    return ICPResult(t,accepted,reason,rmse,len(selected),condition,degenerate,iterations)


def register_scan(source, target, initial=None, max_correspondence_m=.6,
                  max_rmse_m=.18, min_inliers=20, min_normal_eigen_ratio=.01):
    """Point-to-line SE2 ICP for wall scans.

    Sliding along a sampled wall must not be mistaken for zero movement. All
    reliable wall normals participate with Huber weights, retaining distant
    end walls which constrain corridor-longitudinal translation.
    """
    src,dst=_points(source),_points(target)
    transform=np.eye(3) if initial is None else np.array(initial,dtype=float,copy=True)
    if min(len(src),len(dst))<max(min_inliers,8):
        return ICPResult(transform,False,'INSUFFICIENT_POINTS')
    tree=cKDTree(dst)
    _,neighbors=tree.query(dst,k=8)
    local=dst[neighbors];centered=local-local.mean(axis=1,keepdims=True)
    values,vectors=np.linalg.eigh(np.einsum('nki,nkj->nij',centered,centered))
    reliable=(values[:,1]>1e-8)&(values[:,0]<.2*values[:,1])
    normals=vectors[:,:,0]
    for iteration in range(1,31):
        moved=transform_points(transform,src)
        distance,nearest=tree.query(moved)
        keep=(distance<max_correspondence_m)&reliable[nearest]
        if keep.sum()<min_inliers:
            return ICPResult(transform,False,'INSUFFICIENT_WALL_NORMALS')
        points=moved[keep];n=normals[nearest[keep]]
        residual=np.einsum('ij,ij->i',n,points-dst[nearest[keep]])
        normal_values=np.linalg.eigvalsh(n.T@n)
        ratio=float(normal_values[0]/max(normal_values[-1],1e-12))
        if ratio<min_normal_eigen_ratio:
            return ICPResult(transform,False,'DEGENERATE_GEOMETRY',degenerate=True)
        weights=np.sqrt(np.minimum(1.,.05/np.maximum(np.abs(residual),1e-8)))
        jacobian=np.column_stack((n[:,0],n[:,1],-n[:,0]*points[:,1]+n[:,1]*points[:,0]))
        delta,_,rank,_=np.linalg.lstsq(jacobian*weights[:,None],-residual*weights,rcond=1e-8)
        if rank<3 or not np.isfinite(delta).all():
            return ICPResult(transform,False,'DEGENERATE_GEOMETRY')
        # Large updates indicate a wrong local association, never apply them.
        if np.linalg.norm(delta[:2])>.5 or abs(delta[2])>.25:
            return ICPResult(transform,False,'IMPLAUSIBLE_ICP_UPDATE')
        c,s=math.cos(delta[2]),math.sin(delta[2])
        update=np.array([[c,-s,delta[0]],[s,c,delta[1]],[0.,0.,1.]])
        transform=update@transform
        if np.linalg.norm(delta)<1e-5:break
    rmse=float(np.sqrt(np.mean(residual**2)))
    accepted=rmse<=max_rmse_m
    return ICPResult(transform,accepted,'MATCH_OK' if accepted else 'HIGH_RESIDUAL',
                     rmse,int(keep.sum()),1/max(ratio,1e-12),False,iteration)
