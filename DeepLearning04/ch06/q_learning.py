#!/usr/bin/env python

"""
Q-learning の３つの特徴
* TD法
* 方策オフ型
* 重点サンプリングを使わない
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld
from common.utils import greedy_probs


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)  # ターゲット方策
        self.b = defaultdict(lambda: random_actions)  # 挙動方策
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        action_probs = self.b[state]  # 挙動方策から取得
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        self.pi[state] = greedy_probs(self.Q, state, epsilon=0.0)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)


if __name__ == "__main__":
    env = GridWorld()
    agent = QLearningAgent()

    episodes = 10000
    for episode in range(episodes):
        # print(f"episode {episode}")
        state = env.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state, done)
            if done:
                break
            state = next_state

    env.render_q(agent.Q)
