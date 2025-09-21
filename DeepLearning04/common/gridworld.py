#!/usr/bin/env python

import numpy as np
import common.gridworld_render as render_helper


class GridWorld:
    def __init__(self):
        # 行動空間: 行動の候補
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        # 報酬マップ: 各マス目に移動したときに得られる報酬の値
        self.reward_map = np.array(
            [
                [0, 0, 0, 1.0],
                [0, None, 0, -1.0],
                [0, 0, 0, 0],
            ]
        )
        self.goal_state = (0, 3)
        self.wall_state = (1, 1)
        self.start_state = (2, 0)
        self.agent_state = self.start_state

    @property  # メソッドをインスタンス変数として使用出来る
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape

    def actions(self):
        return self.action_space

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def next_state(self, state, action):
        # 1. 移動先の場所の計算
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        # 2. 移動先がグリッドワールドの枠の外か、それとも移動先が壁か?
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state

        # 3. 次の状態を返す
        return next_state

    def reward(self, state, action, next_state):
        return self.reward_map[next_state]

    def reset(self):
        """エージェントをスタート位置に戻す"""
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        """エージェントにactionという行動を行わせ、時間を１ステップ進ませる"""
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = next_state == self.goal_state

        self.agent_state = next_state
        return next_state, reward, done

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(
            self.reward_map, self.goal_state, self.wall_state
        )
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(
            self.reward_map, self.goal_state, self.wall_state
        )
        renderer.render_q(q, print_value)


if __name__ == "__main__":
    env = GridWorld()
    # env.height()ではなく、env.heightとして使用出来る
    print(env.height)  # 3
    print(env.width)  # 4
    print(env.shape)  # (3, 4)

    for action in env.actions():
        print(action)

    print("===")

    for state in env.states():
        print(state)
