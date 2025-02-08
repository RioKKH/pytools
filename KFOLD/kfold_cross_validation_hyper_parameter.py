from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC


class ModelBuilder:
    def __init__(self):
        self.model = SVC()
        #self.model = None

    def build_model(self, params):
        self.model.set_params(**params)
        #self.model = SVC(**params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class DataLoader:
    def __init__(self):
        self.X = None
        self.y = None

    def load_data(self):
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target

    def get_data(self):
        return self.X, self.y


class Pipeline:
    def __init__(self):
        self.data_loader = DataLoader()
        self.model_builder = ModelBuilder()

    def run(self):
        # データの読み込み
        self.data_loader.load_data()
        X, y = self.data_loader.get_data()

        # ハイパーパラメータの探索範囲
        param_grid = {"C": [0.1, 0.5, 1, 5, 10],
                      "gamma": [0.1, 0.5, 1, 5, 10]}

        # K-fold cross validationの設定
        k = 5
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        # GridSearchCVを使用してパラメータの探索とモデルの比較
        grid_search = GridSearchCV(estimator=self.model_builder.model,
                                   param_grid=param_grid, cv=kf)
        grid_search.fit(X, y)

        # 最適なパラメータとモデルの評価結果を表示
        print("Best Parameters:", grid_search.best_params_)
        print("Best Score:", grid_search.best_score_)

        # 最適なモデルを構築
        self.model_builder.build_model(grid_search.best_params_)

        # データをトレーニングデータとテストデータに分割
        train_index, test_index = next(kf.split(X))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # モデルのトレーニング
        self.model_builder.train(X_train, y_train)

        # テストデータで推論
        y_pred = self.model_builder.predict(X_test)
        print("Predictions:", y_pred)


if __name__ == '__main__':
    # パイプラインの実行
    pipeline = Pipeline()
    pipeline.run()
