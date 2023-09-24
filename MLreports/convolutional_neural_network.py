#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from sklearn.metrics import confusion_matrix


class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        #self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 画像のチャンネル数は1で、出力チャンネル数は16、カーネルサイズは3、パディングは1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # 画像のチャンネル数は16で、出力チャンネル数は32、カーネルサイズは3、パディングは1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        #self.relu = nn.ReLU()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # fc1は全結合層
        #self.fc1 = nn.Linear(32 * 14 * 14, 10)
        self.fc1 = nn.Linear(7 * 7 * 32, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        # 入力画像サイズ: [batch_size, 1, 28, 28]
        # １次元目：バッチサイズ
        # ２次元目：チャンネル数
        # ３次元目：縦の画素数
        # ４次元目：横の画素数
        h = self.maxpool(self.relu(self.conv1(x)))
        h = self.maxpool(self.relu(self.conv2(h)))
        h = h.view(h.size(0), -1)
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        h = self.fc3(h)
        return h


class CNNClassifier:

    def __init__(self, epochs=5, learning_rate=0.001, batch_size=64):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = SimpleCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_data\
            = MNIST(root='./data', train=True, download=True, transform=transform)
        self.test_data\
            = MNIST(root='./data', train=False, download=True, transform=transform)

        self.train_loader\
            = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader\
            = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        # 画像・ラベルデータのデータタイプ(型)
        print(type(self.train_data.data), type(self.train_data.targets))
        print(type(self.test_data.data), type(self.test_data.targets))
        # 画像・ラベルの配列サイズ
        print(self.train_data.data.size(), self.train_data.targets.size())
        print(self.test_data.data.size(), self.test_data.targets.size())


    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                # 勾配を0に初期化
                self.optimizer.zero_grad()
                # モデルにデータを入力して予測値を計算
                outputs = self.model(images)
                # 損失関数の計算
                loss = self.criterion(outputs, labels)
                # backwrard()で逆伝播を計算
                loss.backward()
                # パラメータの更新
                self.optimizer.step()
                # 損失の計算
                running_loss += loss.item()
            print(f"Epoch: {epoch+1}, loss: {running_loss/len(self.train_loader)}")


    def evaluate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds==labels).sum().item()

                # .cpu()でテンソルをCPUに移動し、.numpy()でNumPy配列に変換
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

        # concatenate()でリスト内のNumPy配列を結合
        self.all_labels = np.concatenate(all_labels)
        self.all_preds = np.concatenate(all_preds)

        self.class_correct = [0 for _ in range(10)]
        self.class_total = [0 for _ in range(10)]
        for label, pred in zip(self.all_labels, self.all_preds):
            self.class_correct[label] += (label == pred)
            self.class_total[label] += 1

        average_loss = test_loss / total
        accuracy = correct / total

        print(f"Test Loss: {average_loss:.6f}, Test Accuracy: {accuracy:.6f}%")


    def plot_results(self):
        # 各クラスの正答率を計算する
        class_accuracy = [correct / total for correct, total\
                          in zip(self.class_correct, self.class_total)]

        # bar plotで各クラスの正答率を表示する
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        bars = plt.bar(range(10), class_accuracy)
        plt.xlabel('Class label')
        plt.ylabel('Accuracy')
        plt.title('Class-wise accuracy of mnist dataset')
        plt.xticks(self.all_labels)
        plt.ylim(0, 1)

        # 各バーに対応する数値を表示する
        for bar, acc in zip(bars, class_accuracy):
            plt.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() - 0.05,
                     f"{acc:.2f}",
                     ha='center',
                     va='top',
                     color='white',
                     fontsize=12)

        # 混同行列のヒートマップを表示する
        cm = confusion_matrix(self.all_labels, self.all_preds)
        plt.subplot(1, 2, 2)
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion matrix of mnist dataset')

        plt.tight_layout()
        plt.show()


    def run(self):
        self.load_data()
        self.train()
        self.evaluate()
        self.plot_results()
