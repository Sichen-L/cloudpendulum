# -*- coding: utf-8 -*-
"""PENDUBOT controller (shoulder active 0.13 Nm, elbow 5 mNm assist). Hardware run! Requires acados + cloudpendulumclient.

Prerequisite: python ctrl_make_reference.py --plant pendubot (generates a provenance-tagged reference).
Usage:        python ctrl_run_pendubot.py [--t-sim 20] [--no-record]
"""
import sys
from ctrl_run_common import main

if __name__ == "__main__":
    sys.exit(main("pendubot"))
