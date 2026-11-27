"""Small 2-D navigation primitives. Unknown cells and map edges are blocked.

These helpers do not assert that a head-height LiDAR map covers low obstacles;
the ROS adapter requires a verified route and fresh low-obstacle observations.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt, label


def angle_wrap(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


@dataclass
class GridGeometry:
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.resolution,self.origin_x,self.origin_y,self.origin_yaw)) or self.resolution <= 0:
            raise ValueError('invalid map geometry')

    def grid_xy(self, point: Sequence[float]) -> np.ndarray:
        d = np.asarray(point, dtype=float)-[self.origin_x,self.origin_y]
        c,s = math.cos(self.origin_yaw),math.sin(self.origin_yaw)
        return np.array([c*d[0]+s*d[1],-s*d[0]+c*d[1]])/self.resolution

    def cell(self, point: Sequence[float]) -> tuple[int,int]:
        g = self.grid_xy(point)
        return int(math.floor(g[0])),int(math.floor(g[1]))

    def center(self, cell: tuple[int,int]) -> np.ndarray:
        x,y = (np.asarray(cell,dtype=float)+0.5)*self.resolution
        c,s = math.cos(self.origin_yaw),math.sin(self.origin_yaw)
        return np.array([self.origin_x+c*x-s*y,self.origin_y+s*x+c*y])


def inflated_free_grid(occupancy: np.ndarray, resolution: float, radius: float,
                       free_threshold: int = 35) -> np.ndarray:
    """Inflate occupied/unknown cells and borders by a circular footprint."""
    a = np.asarray(occupancy)
    if a.ndim != 2 or min(a.shape) == 0:
        raise ValueError('expected a nonempty 2-D occupancy array')
    if not math.isfinite(resolution) or resolution <= 0 or not math.isfinite(radius) or radius < 0:
        raise ValueError('invalid inflation parameters')
    # Match the map export's free/unknown policy; 36..64 is still uncertain.
    free = (a >= 0) & (a <= free_threshold)
    # An explicit blocked border prevents scipy's all-free-array edge behavior.
    distance = distance_transform_edt(np.pad(free,1,constant_values=False))[1:-1,1:-1]*resolution
    # Occupied cells represent a whole square, not a point at its center.
    return free & (distance > radius + resolution*math.sqrt(2)/2)


def valid_cell(free: np.ndarray, cell: tuple[int,int]) -> bool:
    x,y = cell
    return 0 <= y < free.shape[0] and 0 <= x < free.shape[1] and bool(free[y,x])


def astar(free: np.ndarray, start: tuple[int,int], goal: tuple[int,int],
          max_expansions: int = 500000) -> list[tuple[int,int]]:
    """8-connected A*, with diagonal corner cutting forbidden."""
    if not valid_cell(free,start) or not valid_cell(free,goal):
        raise ValueError('START_OR_GOAL_BLOCKED_OR_OUTSIDE_MAP')
    if start == goal:
        return [start]
    def h(p):return math.hypot(goal[0]-p[0],goal[1]-p[1])
    queue = [(h(start),0.0,start)]
    score,parent,closed = {start:0.0},{},set()
    steps = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    while queue:
        _,cost,current = heapq.heappop(queue)
        if current in closed:continue
        if current == goal:
            path = [goal]
            while path[-1] != start:path.append(parent[path[-1]])
            return path[::-1]
        closed.add(current)
        if len(closed) > max_expansions:raise ValueError('PLANNER_EXPANSION_LIMIT')
        for dx,dy in steps:
            nxt = current[0]+dx,current[1]+dy
            if not valid_cell(free,nxt):continue
            if dx and dy and (not valid_cell(free,(current[0]+dx,current[1]))
                              or not valid_cell(free,(current[0],current[1]+dy))):continue
            tentative = cost+math.hypot(dx,dy)
            if tentative >= score.get(nxt,float('inf')):continue
            score[nxt],parent[nxt] = tentative,current
            heapq.heappush(queue,(tentative+h(nxt),tentative,nxt))
    raise ValueError('NO_COLLISION_FREE_PATH')


def segment_free(free: np.ndarray, geom: GridGeometry, a, b) -> bool:
    """Check all grid squares touched by a segment, including corner neighbors."""
    start,end = geom.grid_xy(a),geom.grid_xy(b)
    delta = end-start
    current = (int(math.floor(start[0])),int(math.floor(start[1])))
    target = (int(math.floor(end[0])),int(math.floor(end[1])))
    on_boundary = [abs(delta[i]) < 1e-14 and abs(start[i]-round(start[i])) < 1e-10
                   for i in (0,1)]
    def traversable(cell):
        x,y=cell
        xs=(x,x-1) if on_boundary[0] else (x,)
        ys=(y,y-1) if on_boundary[1] else (y,)
        return all(valid_cell(free,(cx,cy)) for cx in xs for cy in ys)
    if not traversable(current) or not traversable(target):return False
    # Exact edge/corner endpoints touch both neighboring cells. Keeping this
    # conservative also makes short-cut behavior agree with no-corner-cut A*.
    for point in (start,end):
        touched=[]
        for coordinate in point:
            base=int(math.floor(coordinate))
            touched.append([base,base-1] if abs(coordinate-round(coordinate))<1e-10 else [base])
        if not all(valid_cell(free,(x,y)) for x in touched[0] for y in touched[1]):return False
    step = np.sign(delta).astype(int)
    next_t,increment = [],[]
    for i in (0,1):
        if abs(delta[i]) < 1e-14:
            next_t.append(float('inf'));increment.append(float('inf'))
        else:
            edge = math.floor(start[i])+(1 if step[i]>0 else 0)
            next_t.append((edge-start[i])/delta[i]);increment.append(abs(1/delta[i]))
    limit = abs(target[0]-current[0])+abs(target[1]-current[1])+4
    for _ in range(limit):
        if current == target:return True
        # A crossing at the endpoint is already covered by endpoint checks.
        # Stepping both axes there can otherwise overshoot a negative-going
        # endpoint and loop away from its floor-indexed target cell.
        if min(next_t)>=1.-1e-12:return True
        x,y = current
        if abs(next_t[0]-next_t[1]) < 1e-12:
            neighbors = [(x+int(step[0]),y),(x,y+int(step[1]))]
            if not all(traversable(c) for c in neighbors):return False
            current = x+int(step[0]),y+int(step[1])
            next_t[0]+=increment[0];next_t[1]+=increment[1]
        elif next_t[0] < next_t[1]:
            current = x+int(step[0]),y;next_t[0]+=increment[0]
        else:
            current = x,y+int(step[1]);next_t[1]+=increment[1]
        if not traversable(current):return False
    return False


def plan_path(occupancy: np.ndarray, geom: GridGeometry, start, goal,
              radius: float) -> tuple[list[np.ndarray],np.ndarray]:
    free = inflated_free_grid(occupancy,geom.resolution,radius)
    return _path_on_free(free,geom,start,goal),free


def _path_on_free(free,geom,start,goal):
    cells = astar(free,geom.cell(start),geom.cell(goal))
    raw = [np.asarray(start,dtype=float)]
    raw.extend(geom.center(c) for c in cells[1:-1])
    raw.append(np.asarray(goal,dtype=float))
    # Greedy shortcuts are allowed only if every touched inflated cell is free.
    result,index = [raw[0]],0
    while index < len(raw)-1:
        nxt = len(raw)-1
        while nxt > index+1 and not segment_free(free,geom,raw[index],raw[nxt]):nxt-=1
        if not segment_free(free,geom,raw[index],raw[nxt]):raise ValueError('INVALID_PATH_SEGMENT')
        result.append(raw[nxt]);index=nxt
    return result


def plan_observed_path(occupancy,geom,start,goal,radius):
    """Advance through observed free space while a distant goal is still unknown."""
    free=inflated_free_grid(occupancy,geom.resolution,radius)
    start_cell=geom.cell(start)
    if not valid_cell(free,start_cell):
        raise ValueError('WAIT_OBSERVED_START_FOOTPRINT')
    components,_=label(free)
    reachable=components==components[start_cell[1],start_cell[0]]
    if valid_cell(reachable,geom.cell(goal)):
        return _path_on_free(free,geom,start,goal),free
    # Never reinterpret an observed occupied goal as an exploration target.
    gx,gy=geom.cell(goal)
    if (0<=gy<occupancy.shape[0] and 0<=gx<occupancy.shape[1]
            and occupancy[gy,gx]>=65):raise ValueError('GOAL_OCCUPIED')
    start=np.asarray(start,dtype=float);goal=np.asarray(goal,dtype=float)
    delta=goal-start;distance=float(np.linalg.norm(delta))
    if distance<.4:raise ValueError('GOAL_NOT_OBSERVED_FREE')
    for advance in np.linspace(min(2.,distance),.4,9):
        target=start+delta*(advance/distance)
        if not valid_cell(reachable,geom.cell(target)):continue
        try:return _path_on_free(free,geom,start,target),free
        except ValueError:continue
    raise ValueError('NO_OBSERVED_FREE_PROGRESS_TOWARD_GOAL')


def waypoint_command(pose, waypoint, goal, max_vx=0.15, max_wz=0.35, min_vx=0.0):
    x,y,yaw = pose
    dx,dy = np.asarray(waypoint)-[x,y]
    error = angle_wrap(math.atan2(dy,dx)-yaw)
    distance = float(np.linalg.norm(np.asarray(goal)-[x,y]))
    wz = float(np.clip(1.4*error,-max_wz,max_wz))
    if abs(error) > 0.6:return 0.0,wz
    vx = max_vx*max(0.,math.cos(error))*min(1.,distance/0.60)
    vx = min(max_vx,max(vx,min_vx))
    return float(vx),wz


def limit_command(previous, desired, dt, max_accel=.15, max_angular_accel=.4):
    if dt <= 0:return tuple(previous)
    bound = np.array([max_accel,max_angular_accel])*min(dt,.2)
    value = np.asarray(previous)+np.clip(np.asarray(desired)-previous,-bound,bound)
    return float(value[0]),float(value[1])


def swept_collision(points_xy: np.ndarray, pose, vx: float, wz: float, radius: float,
                    stop_time: float = 1.0, deceleration: float = .15) -> bool:
    """Conservative disk/capsule envelope for a short forward/turn command."""
    points = np.asarray(points_xy,dtype=float)
    if points.size == 0:return False
    points = points.reshape(-1,2)
    center = np.asarray(pose[:2],dtype=float)
    heading = np.array([math.cos(pose[2]),math.sin(pose[2])])
    length = max(0.0,vx)*stop_time+vx*vx/(2*max(deceleration,1e-3))
    d = points-center
    along = np.clip(d@heading,0.,length)
    delta = d-along[:,None]*heading
    return bool(np.any(np.einsum('ij,ij->i',delta,delta) <= radius*radius))


class ArrivalGate:
    def __init__(self,tolerance=.20,settle_sec=1.,linear_speed=.03,angular_speed=.05):
        self.tolerance,self.settle_sec=tolerance,settle_sec
        self.linear_speed,self.angular_speed=linear_speed,angular_speed
        self.since=None

    def update(self,now,distance,linear_speed,angular_speed):
        inside = distance <= self.tolerance and abs(linear_speed)<=self.linear_speed and abs(angular_speed)<=self.angular_speed
        if not inside:self.since=None;return False
        if self.since is None or now < self.since:self.since=now
        return now-self.since >= self.settle_sec
