#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)  # 各マシンの勝率

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


class Agent:
    """
    Attributes:
        epsilon (float): ε-greedy法におけるランダムに行動する確率
        Qs (list[float]): 行動価値
    """

    def __init__(self, epsilon, action_size=10):
        """
        Args:
            action_size (int): エージェントが選択できる
        """
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        """スロットマシンの価値を推定するメソッド"""
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        """ε-greedy法に基づいて行動を選択するメソッド"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


def main():
    runs = 200
    steps = 1000
    epsilon = 0.1
    all_rates = np.zeros((runs, steps))  # (200, 1000) の形状の配列

    for run in range(runs):  # 200回の実験
        bandit = Bandit()
        agent = Agent(epsilon)
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()  # 1. 行動を選ぶ
            reward = bandit.play(action)  # 2. 実際にプレイして報酬を得る
            agent.update(action, reward)  # 3. 行動と報酬から学ぶ
            total_reward += reward
            rates.append(total_reward / (step + 1))

        all_rates[run] = rates  # 1. 報酬の結果を記録する

    avg_rates = np.average(all_rates, axis=0)  # 2. 各ステップにおける平均を求める

    # グラフの描画
    plt.ylabel("Rates")
    plt.xlabel("Steps")
    plt.plot(avg_rates)
    plt.show()


if __name__ == "__main__":
    main()
