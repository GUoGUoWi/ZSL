import logging
from collections import defaultdict
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from data.modelnet_dataset import MultiViewDataset


class EnsemblePredictor:
    def __init__(self, clients, device='cuda'):
        self.clients = clients
        self.device = device
        self.weights = None

    def _evaluate_client(self, client, test_dataset):
        """
        评估单个客户端模型的性能
        """
        # 创建数据加载器
        test_loader = DataLoader(
            MultiViewDataset(test_dataset['X'], test_dataset['Y']),
            batch_size=32,
            shuffle=False
        )

        client.base_model.eval()
        correct = 0
        total = 0

        # 每个类的正确预测统计
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                batch_X = [X.to(self.device) for X in batch_X]
                batch_Y = batch_Y.to(self.device)

                outputs = client.base_model(batch_X, mode='client')
                _, predicted = torch.max(outputs.data, 1)

                total += batch_Y.size(0)
                correct += (predicted == batch_Y).sum().item()

                # 统计每个类的准确率
                for label, prediction in zip(batch_Y, predicted):
                    class_total[label.item()] += 1
                    if label == prediction:
                        class_correct[label.item()] += 1

        # 计算整体准确率
        overall_accuracy = 100 * correct / total

        # 计算每个类的准确率
        class_accuracies = {cls: 100 * class_correct[cls] / class_total[cls]
        if class_total[cls] > 0 else 0.0
                            for cls in class_total.keys()}

        return overall_accuracy, class_accuracies

    def validate_and_set_weights(self, val_dataset):
        """
        在验证集上评估每个客户端并设置权重
        """
        accuracies = []
        for client in self.clients:
            val_acc, _ = self._evaluate_client(client, val_dataset)
            accuracies.append(val_acc)

        # 使用softmax将准确率转换为权重
        accuracies = torch.tensor(accuracies)
        weights = F.softmax(accuracies / 10.0, dim=0).tolist()  # temperature=10.0
        self.weights = weights
        return weights

    def predict(self, test_dataset):
        """
        对测试数据进行集成预测
        """
        if self.weights is None:
            self.weights = [1.0 / len(self.clients)] * len(self.clients)

        test_loader = DataLoader(
            MultiViewDataset(test_dataset['X'], test_dataset['Y']),
            batch_size=32,
            shuffle=False
        )

        all_predictions = []
        true_labels = []

        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                batch_X = [X.to(self.device) for X in batch_X]
                batch_predictions = []

                # 获取每个客户端的预测
                for client in self.clients:
                    client.base_model.eval()
                    outputs = client.base_model(batch_X, mode='client')
                    probs = F.softmax(outputs, dim=1)
                    batch_predictions.append(probs)

                # 加权组合预测结果
                ensemble_pred = torch.zeros_like(batch_predictions[0])
                for pred, weight in zip(batch_predictions, self.weights):
                    ensemble_pred += weight * pred

                all_predictions.append(ensemble_pred)
                true_labels.append(batch_Y)

        return torch.cat(all_predictions), torch.cat(true_labels)

    def evaluate(self, test_dataset):
        """
        评估集成模型的性能
        """
        predictions, labels = self.predict(test_dataset)
        _, predicted = torch.max(predictions, 1)
        labels = labels.to(self.device)

        # 计算整体准确率
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        overall_accuracy = correct / total

        # 计算每个类的准确率
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        for label, prediction in zip(labels, predicted):
            class_total[label.item()] += 1
            if label == prediction:
                class_correct[label.item()] += 1

        class_accuracies = {cls: class_correct[cls] / class_total[cls]
        if class_total[cls] > 0 else 0.0
                            for cls in class_total.keys()}

        return overall_accuracy, class_accuracies