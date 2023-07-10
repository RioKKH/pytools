"""
k-fold cross validationではデータセットを複数の部分集合に分割し、
それぞれを交差検証に使用する。この手法を使用することにより、以下の
利点がある。

１．モデルの汎化性能の評価
データセット全体をトレーニングデータとテストデータに分割する代わりに、
k回の分割を行う事で、モデルの汎化性能をより正確に評価することが出来る。
すべてのデータがトレーニングに使用され、すべてのデータがテストに使用
されるため、モデルのパフォーマンスの偏りを防ぐことが出来る。

２．パラメータの調整とモデルの選択
k-fold cross validationでは、複数のトレーニングデータセットとテスト
データセットの組み合わせでモデルを評価するため、ハイパーパラメータの
チューニングや異なるモデルの比較が容易になる。これにより、最適な
パラメータ設定は最適なモデルの選択を行うことが出来るようになる。
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

# Irisデータセットの読み込み
iris = load_iris()
X = iris.data
y = iris.target

# K-fold cross validationの設定
k = 5 # 分割数
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# モデルの設定
model = KNeighborsClassifier(n_neighbors=3)

# K-fold cross validationの実行
fold = 1
for train_index, test_index in kf.split(X):
    print("Fold:", fold)
    print("Training samples:", len(train_index))
    print("Testing samples:", len(test_index))

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # モデルの学習
    model.fit(X_train, y_train)

    # テストデータの評価
    accuracy = model.score(X_test, y_test)
    print("Accuracy:", accuracy)

    fold += 1
    print("-------------------------")


