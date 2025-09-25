#!/usr/bin/env python

if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
from common.gridworld import GridWorld
from ch04.policy_eval import policy_eval


def argmax(d):
    """d (dict)"""
    max_value = max(d.values)
    max_key = -1
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


def greedy_policy(V, env, gamma):
    pi = {}

    for state in env.states():
        action_values = {}
