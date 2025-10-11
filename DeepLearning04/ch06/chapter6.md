### 6章

* TD法：
  * 環境のモデル（状態遷移確率、報酬関数）を使わない。
  * エピソードの終わりを待つことなく、行動を一つ行うたびに価値関数を更新する。

* SARSA

  1. 特徴

     * (Off-policyな実装もあるが、一般には)**On-policy**手法
       * 実際に行動する方策と学習する方策が同じ
       * 探索的な行動も含めて学習する

  2. 実用上の特徴

     * より保守的で安全な方策を学習する

     * 探索中のリスクも考慮する
       * 崖に近い経路は、探索中に落ちるリスクがあるために避ける

  3. 更新式

     1. ベルマン方程式　→　SARSA
        次の状態$S_{t+1}$で実際に選択した行動$a_{t+1}$のQ値を使う

  $$
  Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]
  $$

* Q学習

  1. 特徴

     * 実際に行動する方策と学習する方策が異なる。**Off-policy**手法。
       * 挙動方策とターゲット方策

     * 探索的な行動をしても最適な行動を仮定して学習

  2. 実用上の特徴

     * 寄り探索的で最適な方策を学習する

     * 理論的には最適方策に収束する
       * 最短経路が崖沿いなら、探索中のリスクは無視してその経路を学習する

  3. 更新式

     1. ベルマン最適方程式　→　Q学習
        次の状態$S_{t+1}$で最大のQ値を持つ行動を使う(max演算子)

  $$
  Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t)]
  $$

#### 6.5 分布モデルとサンプルモデル

##### 6.5.1 分布モデルとサンプルモデル

エージェントの実装方法（環境に関しても分布モデルとサンプルモデルがある）。

* 分布モデル：確率分布を明示的に保持する
  ```python
  from collections import defaultdict
  import numpy as np
  
  class RandomAgent:
      def __init__(self):
          random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25} # 確率分布
          self.pi = defaultdict(lambda: random_actions)
          
      def get_action(self, state):
          action_probs = self.pi[state]
          actions = list(action_probs.keys())
          probs = list(action_probs.values())
          return np.random.choice(actions, p=probs) # サンプリング
  ```

  

* サンプルモデル：サンプリング出来ることだけが条件
  ```python
  class RandomAgent:
      def get_action(self, state):
          return np.random.choice(4)
  ```

  

