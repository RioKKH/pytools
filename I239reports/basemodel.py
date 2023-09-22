#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

class BaseModel(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_data(self):
        """データセットをロードし、訓練データとテストデータに分割する"""
        pass

    @abstractmethod
    def preprocess_data(self, X_train, X_test):
        """データの前処理を行う"""
        pass

    @abstractmethod
    def build_model(self):
        """モデルを構築する"""
        pass

    @abstractmethod
    def train_model(self, X_train, y_train):
        """モデルを訓練する"""
        pass

    @abstractmethod
    def evaluate_model(self, X_test, y_test):
        """モデルを評価する"""
        pass

    @abstractmethod
    def save_results(self):
        """モデルや評価結果を保存する"""
        pass

    @abstractmethod
    def plot_results(self):
        """モデルや評価結果をプロットする"""
        pass


