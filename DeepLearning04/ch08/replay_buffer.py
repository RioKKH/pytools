#!/usr/bin/env python

from collections import deque
import random
import numpy as np
import gym


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """経験データを追加するメソッド"""
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        # stateは形状が(4,)のnp.ndarray
        # np.stack([s7, s2, s5])によって (3, 4)のnp.ndarrayになる
        state = np.stack([x[0] for x in data])
        # actionはint型
        # np.stack([a7, a2, a5])によって (3,)のnp.ndarrayになる
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)

    for episode in range(10):
        state = env.reset()
        done = False

        while not done:
            env.render()
            action = 0
            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
    env.close()

    state, action, reward, next_state, done = replay_buffer.get_batch()
    print(state.shape)  # (32, 4)
    print(action.shape)  # (32,)
    print(reward.shape)  # (32,)
    print(next_state.shape)  # (32, 4)
    print(done.shape)  # (32,)
