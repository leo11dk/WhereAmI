#!/usr/bin/env python3
"""Run Mission 1 numerical and protocol checks without starting ROS or motors."""

from pathlib import Path
import sys
import unittest


def main():
    custom = Path(__file__).resolve().parents[1]
    loader = unittest.TestLoader()
    suites = []
    for package in ('haic_m1_sensors', 'haic_m1_navigation'):
        root = custom / package
        sys.path.insert(0, str(root))
        suites.append(loader.discover(str(root / 'tests'), top_level_dir=str(root / 'tests')))
    suites.append(loader.discover(str(custom / 'tests'), top_level_dir=str(custom / 'tests')))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suites))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
