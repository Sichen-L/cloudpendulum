# -*- coding: utf-8 -*-
"""ACROBOT controller (elbow active 0.13 Nm, shoulder 5 mNm assist). Hardware run! Requires acados + cloudpendulumclient.

Prerequisite: python ctrl_make_reference.py --plant acrobot (generates a provenance-tagged reference).
Usage:        python ctrl_run_acrobot.py [--t-sim 20] [--no-record]
"""
import sys
from ctrl_run_common import main

if __name__ == "__main__":
    sys.exit(main("acrobot"))
