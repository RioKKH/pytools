### 10章 さらに先へ

#### 10.1 深層強化学習アルゴリズムの分類

- モデルベース
  環境のモデル 状態遷移関数$p(s'|s, a)$と報酬関数$r(s, a, s')$を使用するもの
  - モデルが既知
    ボードゲームでは、環境のモデルが既知の問題設定として扱う事が出来る。
    - AlphaGo
    - AlphaZero
  - モデルを学習
    環境から得た経験によって環境のモデルを学習する。
    - World Models
    - MBVE (Model-Based Value Estimation)
- モデルフリー
  2022年の段階ではモデルフリーの手法の方が成果をあげている。
  - 方策ベース
    - 方策勾配法
    - REINFORCE
    - A3C, A2C
      分散学習を行うアルゴリズム
    - DDPG
      決定論的な方策を持つアルゴリズム
    - TRPO, PPO
      目的関数に制約を追加するアルゴリズム
  - 価値ベース
    - DQN
    - Double DQN
  - 両方を用いるもの
    - Actor-Critic

#### 10.2 方策勾配法系列の発展アルゴリズム

##### 10.2.1 A3C, A2C

- A3C
  - Actor-Criticを使って方策を学習
  - 1つのグローバルネットワークと複数のローカルネットワークを使う。
  - 定期的にグローバルネットワークとローカルネットワークの重みパラメータを同期する
    - 更新は非同期
  - 経験再生：方策ON型の手法では使えない
  - 並列処理：方策ON型の手法でも使える！
  - 各環境でニューラルネットワークを実行する櫃ようがある。従って、理想的にはＮ個の環境に対してＮ個のGPUが必要となる。
- A2C
  - パラメータを同期的に更新する。
  - ニューラルネットワークを実行する箇所が１箇所にまとめられる→GPUが１つでも計算することが可能。

##### 10.2.2 DDPG Deep Deterministic Policy Gradient method

連続的な行動空間の問題に対して設計されたアルゴリズム

##### 10.2.3 TRPO, PPO

- TRPO Trust Region Policy Optimization
  - 適切なステップ幅で方策を最適化する
    - ２つの確率分布がどれだけ似ているかを計測する使用としてKLダイバージェンスを用いる
  - ヘッセ行列を計算する必要があり、計算量の多さがボトルネック
- PPO Proximal Policy Optimization

#### 10.3 DQN系列の発展アルゴリズム

##### 10.3.1 カテゴリカルDQN

##### 10.3.2 Noisy Network

##### 10.3.3 Rainbow

- 以下の手法を全て組み合わせたDQN
  - Double DQN
  - 優先度付き経験再生
  - Dueling DQN
  - カテゴリカル DQN
  - Noisy Network

##### 10.3.4 Rainbow以降の発展アルゴリズム

- Ape-X: Rainbow + 複数エージェントを別CPUで実行 + 異なる探索率ε
- R2D2: Recurrent + Replay + Distributed + Deep Q-Network
- NGU: R2D2 + 内発的報酬 (intrinsic reward)
- Agent57: NGU + メタコントローラ→Atariの57のゲーム全てで初めて人間よりも良い成績を修めた