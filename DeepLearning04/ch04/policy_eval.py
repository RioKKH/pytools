#!/usr/bin/env python

if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
from common.gridworld import GridWorld


def eval_onestep(pi, V, env, gamma=0.9):
    # 各状態へアクセスする
    for state in env.states():
        # ゴールの価値関数は常に0
        # --> エージェントがゴールにいるということはそこでエピソードは終了
        if state == env.goal_state:
            V[state] = 0
            continue

        # probsはprobabilitiesの略
        action_probs = pi[state]
        new_V = 0
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            # 新しい価値関数
            new_V += action_prob * (r + gamma * V[next_state])
        V[state] = new_V
    return V


def policy_eval(pi, V, env, gamma, thereshold=0.001):
    while True:
        # 更新前の価値関数
        old_V = V.copy()
        print("Call eval_onestep()")
        V = eval_onestep(pi, V, env, gamma)

        # 更新された量の最大値を求める
        delta = 0
        for state in V.keys():
            print(state)
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        # 閾値との比較
        if delta < thereshold:
            print("Break!")
            break
    return V


if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9

    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    V = policy_eval(pi, V, env, gamma)
    print("before render_v")
    env.render_v(V, pi)
    print("after render_v")
