"""Runtime settings shared by launch and navigation; no ROS imports required."""

import json
import math
from pathlib import Path


NAVIGATION_DEFAULTS = dict(
    online_mapping=False,
    allow_motion=False, footprint_verified=False, route_verified=False, ground_verified=False,
    lidar_floor_z_m=-1.3, obstacle_min_height_m=.10, obstacle_max_height_m=1.8,
    robot_radius_m=.35, safety_margin_m=.15, max_vx=.15, max_wz=.35,
    max_accel=.15, max_angular_accel=.4, goal_tolerance_m=.20, settle_sec=1.,
    max_pose_age_sec=.5, input_timeout_sec=1., max_wall_age_sec=3.,
    goal_timeout_sec=180., stall_timeout_sec=15., obstacle_timeout_sec=10.,
    cloud_point_budget=200000, min_scan_hits=20,
)


def chapter_nodes(chapter):
    if type(chapter) is not int or chapter not in range(1, 7):
        raise ValueError('chapter must be an integer from 1 to 6')
    active = [] if chapter == 1 else ['odometry']
    if chapter == 3:
        active.append('mapping')
    if chapter >= 4:
        active.extend(['localization', 'alignment'])
    if chapter >= 5:
        active.append('navigation')
    return active


def load_runtime(path):
    """Reject misspelled keys and JSON coercions before starting any nodes."""
    settings = json.loads(Path(path).expanduser().read_text())
    if not isinstance(settings, dict):
        raise ValueError('m1_runtime.json must contain an object')
    defaults = {'auto_seed_on_start': False, **NAVIGATION_DEFAULTS}
    unknown = set(settings) - defaults.keys()
    if unknown:
        raise ValueError('Unknown m1_runtime.json settings: ' + ', '.join(sorted(unknown)))
    for key, value in settings.items():
        expected = type(defaults[key])
        if expected is bool:
            valid = type(value) is bool
        elif expected is int:
            valid = type(value) is int and value > 0
        else:
            valid = type(value) in (int, float) and math.isfinite(value)
        if not valid:
            raise ValueError(f'Invalid m1_runtime.json value for {key}: {value!r}')
        if expected is float:
            settings[key] = float(value)
    return {**defaults, **settings}
