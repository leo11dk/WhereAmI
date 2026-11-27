#!/usr/bin/env python3
"""Run after sourcing the installed haic_m1_navigation overlay."""
import argparse,json
from haic_m1_navigation.alignment_core import calibrate_file

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True)
    p.add_argument('--landmarks',required=True)
    p.add_argument('--confirm-landmarks',action='store_true')
    a=p.parse_args()
    print(json.dumps(calibrate_file(a.data_dir,a.landmarks,a.confirm_landmarks),indent=2))
