#!/usr/bin/env python3
"""Practice one supplied world goal with the official runner and referee."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goal', type=float, nargs=2, required=True, metavar=('X', 'Y'))
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--no-map', action='store_true', help='Keep the simulator window without opening RViz')
    parser.add_argument('--timeout', type=float, default=360.)
    args = parser.parse_args(argv)
    x, y = args.goal
    if not (math.isfinite(x) and math.isfinite(y)):
        parser.error('Goal must contain finite world coordinates X Y')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be positive and finite')
    src = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, HAIC_M1_CHAPTER='6', PYTHONDONTWRITEBYTECODE='1')
    environment['HAIC_M1_SHOW_MAP'] = 'false' if args.headless or args.no_map else 'true'
    # This is a practice scenario CREATED from the CLI goal. Mission code never
    # opens a scenario or reads hidden targets; the official runner owns it.
    with tempfile.TemporaryDirectory(prefix='haic_m1_practice_') as directory:
        directory = Path(directory)
        scenario = directory / 'goal.yaml'
        scenario.write_text(json.dumps({'m1_goal': [x, y, 0.]}))
        config = directory / 'config.yaml'
        simulator = {'view': not args.headless}
        if args.headless:
            simulator.update(shadow_size=2048, extra_args=[
                '--camera-hz', '0', '--panda-camera-hz', '0', '--no-panda'])
        config.write_text(json.dumps({'simulator': simulator}))
        command = [sys.executable, str(src/'main.py'), '--missions', '1', '--no-wait',
                   '--no-step', '--scenario', str(scenario), '--config', str(config)]
        print(f'Mission 1 practice: world goal ({x}, {y}); official referee radius 0.5m', flush=True)
        child = subprocess.Popen(command, cwd=src, env=environment, start_new_session=True)
        try:
            return child.wait(timeout=args.timeout)
        except (KeyboardInterrupt, subprocess.TimeoutExpired) as exc:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
            return 130 if isinstance(exc, KeyboardInterrupt) else 124


if __name__ == '__main__':
    raise SystemExit(main())
