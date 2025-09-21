#!/usr/bin/env python

if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from collections import defaultdict
    from common.gridworld import GridWorld

    def eval_onestep(pi, V, env, gamma=0.9):
        for state in env.states():

