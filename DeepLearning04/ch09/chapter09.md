### 9章 方策勾配法

ここまで価値ベースの手法を扱っていた (Value-based Method)。Q関数や状態価値関数の事を指す。価値関数をモデル化し、価値関数を学習する手法。価値関数を経由して方策を得る。

一般化方策反復：価値関数の評価と方策を改善するプロセスを反復する事で、最適方策に近づく

価値関数を経由せずに方策を直接表す手法の事を方策ベースの手法(Policy-based Method)という。

この章では

- 単純な方策勾配法
- REINFORCE
- ベースライン付きREINFORCE
- Actor-Critic

の手法を確認する。

#### 9.1 最も単純な方策勾配法

方策 $\pi(a|s)$：状態sにおいてaという行動をとる確率。ニューラルネットワークでモデル化する。
ニューラルネットワークの全ての重みを${\theta}$という記号に集約する。この時の方策を$\pi_{\theta}(a|s)$と表現する。

方策$\pi_{\theta}$を使って目的関数を設定する。目的関数を最大化する$\theta$を見つける。

- 目的関数
  $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$
- 勾配(方策勾配定理)
  $\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]$
- パラメータ更新
  $\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)$

#### 9.2 REINFORCE

REward Increment = Nonnegative Factor x Offset Reinforcement x Characteristic Eligibility

- 収益(リターン)
  $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$
- 勾配
  $\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$
- パラメータ更新
  $\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$

#### 9.3 ベースライン

ある結果に対して予測値を引くことで分散を減らす事が出来る。これによって学習を安定させることが出来る。実践的にはベースライン$b(S_t)$ としては、価値関数が用いられる。$b(S_t)=V_{\pi_{\theta}}(S_t)$となる。ベースラインを使って分散を小さくすることが得きれば、サンプル効率の良い学習が行える。

- アドバンテージ
  $A_t = G_t - V(s_t)$
- 勾配
  $\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - V(s_t))\right]$
- 方策パラメータ更新
  $\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - V(s_t))$ 
- 価値関数更新
  $V(s_t) \leftarrow V(s_t) + \beta (G_t - V(s_t))$

#### 9.4 Actor-Critic

- TD誤差(アドバンテージの推定)
  $\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)$
- Actor(方策)の更新
  $\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t$
- Critic(価値関数)の更新
  $w \leftarrow w + \beta \delta_t \nabla_w V_w(s_t)$
- 勾配降下法の形式で
  $w \leftarrow w - \beta \nabla_w \frac{1}{2}\delta_t^2$

#### ９章で学ぶアルゴリズムのまとめ

| アルゴリズム              | 評価指標                        | 主な改善点             |
| ------------------------- | ------------------------------- | ---------------------- |
| シンプル方策勾配法        | エピソード全体の報酬 $R(\tau) $ | 基本形                 |
| REINFORCE                 | 時刻 $t $ 以降の収益 $G_t $     | 因果性を考慮           |
| ベースライン付きREINFORCE | アドバンテージ $G_t - V(s_t) $  | 分散の削減             |
| Actor-Critic              | TD誤差 $\delta_t $              | オンライン学習、高速化 |