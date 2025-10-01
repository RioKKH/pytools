#!/usr/bin/env python

# __file__が定義されている環境でのみパス走査を実行し、NameErrorを防ぐ
if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
from common.gridworld import GridWorld
from ch04.policy_iter import greedy_policy


def value_iter_onestep(V, env, gamma):
    for state in env.states():  # 全ての状態にアクセスする
        if state == env.goal_state:  # ゴールの価値関数は常に0
            V[state] = 0
            continue

        action_values = []
        for action in env.actions():  # 全ての行動にアクセスする
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]  # 新しい価値関数
            action_values.append(value)

        V[state] = max(action_values)  # 最大値を取り出す
    return V


def value_iter(V, env, gamma, threshold=0.001, is_render=True):
    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()  # 更新前の価値関数
        V = value_iter_onestep(V, env, gamma)

        # 更新された量の最大値を求める
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        # 閾値との比較
        if delta < threshold:
            break
    return V


if __name__ == "__main__":
    V = defaultdict(lambda: 0)  # 価値関数の初期化
    env = GridWorld()
    gamma = 0.9

    V = value_iter(V, env, gamma)

    pi = greedy_policy(V, env, gamma)
    env.render_v(V, pi)
