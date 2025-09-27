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
    max_value = max(d.values())
    max_key = -1
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


def greedy_policy(V, env, gamma):
    pi = {}

    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values[action] = value

        max_action = argmax(action_values)
        action_probs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        action_probs[max_action] = 1.0
        # greedy化した方策を返す
        pi[state] = action_probs
    return pi


def policy_iter(env, gamma, threshold=0.001, is_render=True):
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    while True:
        # 1. 現状の方策を評価
        V = policy_eval(pi, V, env, gamma, threshold)
        # 2. 改善 (greedy化した方策new_piを得る
        new_pi = greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, pi)

        # 3. 更新確認
        # もし更新されていなければ、ベルマン最適方程式を満たしているということ。
        # その時のpi (とnew_pi)が最適方策ということになる
        if new_pi == pi:
            break
        pi = new_pi

    return pi


if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9
    pi = policy_iter(env, gamma)
