import logging
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.CRNet.models_cls import CRnetCls
from models.CRNet.train_CRNet_cls import train_and_eval_CRNet
from models.UMLModel import UMLFramework
from models.base_models import MLP3

from tqdm import tqdm

global TORCH_BACKEND_DEVICE  # torch 后端设备，可能是cpu、cuda、mps
if torch.cuda.is_available():
    TORCH_BACKEND_DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    TORCH_BACKEND_DEVICE = torch.device('mps')
else:
    TORCH_BACKEND_DEVICE = torch.device("cpu")

# 临时用，模拟生成的数据
def sample_dataset_by_class(features, labels, samples_per_class=100):
    """
    从每个类别中各抽取指定数量的样本，形成新的数据集。

    参数:
    features (torch.Tensor): 数据集的特征张量。
    labels (torch.Tensor): 数据集的标签张量。
    samples_per_class (int): 每个类别抽取的样本数量。

    返回:
    new_features (torch.Tensor): 抽取后的特征张量。
    new_labels (torch.Tensor): 抽取后的标签张量。
    """
    unique_classes = labels.unique()
    sampled_features = []
    sampled_labels = []

    for cls in unique_classes:
        # 获取当前类别的索引
        cls_indices = (labels == cls).nonzero(as_tuple=True)[0]
        # 随机抽取指定数量的样本
        sampled_indices = np.random.choice(cls_indices, samples_per_class, replace=False)
        # 收集抽取的样本
        sampled_features.append(features[sampled_indices])
        sampled_labels.append(labels[sampled_indices])

    # 将列表转换为张量
    new_features = torch.cat(sampled_features, dim=0)
    new_labels = torch.cat(sampled_labels, dim=0)

    # 打乱样本
    perm = torch.randperm(new_features.size(0))
    new_features = new_features[perm]
    new_labels = new_labels[perm]

    new_dataset = {'X': new_features, 'Y': new_labels}

    return new_dataset


class VFLClient:
    def __init__(self, class_dict, unaligned_dataset, aligned_dataset, generated_dataset, epochs=40,
                 num_tasks=20, table_dim=7, num_classes=6, client_id=0, cluster_num=4, meta_params:dict=None,
                 label_mode='sup', test_dataset=None):
        """
        初始化客户端对象
        :param class_dict: 分类字典
        :param unaligned_dataset: 未对齐数据集
        :param aligned_dataset: 对齐数据集
        :param generated_dataset: LLM生成数据集
        :param epochs: 训练总轮数
        :param num_tasks: 元训练轮数
        :param table_dim: 模型输入维度
        :param num_classes: 分类数量
        :param meta_params: 元学习参数
        """
        self.client_id = client_id
        self.base_model = MLP3(
            input_dim=table_dim)
        print(self.base_model)
        self.device = TORCH_BACKEND_DEVICE
        if meta_params is None:
            self.model = UMLFramework(self.base_model, class_dict=class_dict, num_tasks=num_tasks, input_dim=table_dim).to(self.device)
        else:
            print(f"Using modified meta_params, {meta_params}")
            N_way = meta_params['N_way']
            K_shot = meta_params['K_shot']
            Q_query = meta_params['Q_query']
            inner_steps = meta_params['inner_steps']
            self.model = UMLFramework(self.base_model, class_dict=class_dict, num_tasks=num_tasks,
                                      N_way=N_way, K_shot=K_shot, input_dim=table_dim,
                                      Q_query=Q_query, inner_step=inner_steps).to(self.device)
        self.unaligned_dataset = unaligned_dataset
        self.aligned_dataset = aligned_dataset
        self.test_dataset = test_dataset
        self.pseudo_label = None
        self.label_mode = label_mode
        self.cluster_num = cluster_num
        if self.label_mode != 'sup':
            print('generate pseudo label')
            self.generate_pseudo_labels_for_tabular()
        # TODO: 分离生成数据集和对齐数据集
        self.generated_dataset = sample_dataset_by_class(generated_dataset['X'], generated_dataset['Y'])
        self.epochs = epochs
        self.CRNet = CRnetCls(num_experts=3, semantic_dim=200, visual_dim=16, hidden_dim=100).to(self.device)

    def meta_learning_old(self):
        # 合并未对齐数据和对齐数据
        combined_X = torch.cat((self.unaligned_dataset['X'], self.aligned_dataset['X']), dim=0)
        combined_Y = torch.cat((self.unaligned_dataset['Y'], self.aligned_dataset['Y']), dim=0)
        generated_X = self.generated_dataset['X']
        generated_Y = self.generated_dataset['Y']
        # 创建新的数据集
        combined_dataset = TensorDataset(combined_X, combined_Y)
        # 将数据集划分为训练集和测试集
        train_size = int(0.8 * len(combined_dataset))  # 80%用于训练
        test_size = len(combined_dataset) - train_size  # 20%用于测试
        train_dataset, test_dataset = torch.utils.data.random_split(combined_dataset, [train_size, test_size])
        # 创建训练集和测试集的数据加载器
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        # 创建生成数据集的数据加载器
        generated_dataset = TensorDataset(generated_X, generated_Y)
        generated_loader = DataLoader(generated_dataset, batch_size=128, shuffle=True)
        for epoch in tqdm(range(self.epochs)):
            # 训练模型
            self.model.train_model_old(train_loader=train_loader, test_loader=test_loader, generated_loader=None, epoch=epoch)
        for epoch in tqdm(range(self.epochs)):
            # 微调模型
            self.model.fine_tune_old(train_loader=train_loader, test_loader=test_loader, generated_loader=None, epoch=epoch)

    def meta_learning(self, load=True):
        # 创建模型保存目录
        model_dir = os.path.join('models', 'base_model_dict')
        os.makedirs(model_dir, exist_ok=True)

        # 根据 client_id 生成模型文件名
        model_filename = f"{self.client_id}_base_model.pth"
        model_path = os.path.join(model_dir, model_filename)

        # 检查是否存在保存的模型
        if os.path.exists(model_path) and load:
            print(f"Loading existing model for client {self.client_id}...")
            self.model.load_state_dict(torch.load(model_path))
            print("Model loaded successfully.")
            return True
        else:
            print(f"No existing model found for client {self.client_id}. Starting with a new model.")

        # 合并未对齐数据和对齐数据
        combined_X = torch.cat((self.unaligned_dataset['X'], self.aligned_dataset['X']), dim=0)
        combined_Y = torch.cat((self.unaligned_dataset['Y'], self.aligned_dataset['Y']), dim=0)
        generated_X = self.generated_dataset['X']
        generated_Y = self.generated_dataset['Y']

        # 创建新的数据集
        combined_dataset = TensorDataset(combined_X, combined_Y)

        # 将数据集划分为训练集和测试集
        train_size = int(0.8 * len(combined_dataset))  # 80%用于训练
        test_size = len(combined_dataset) - train_size  # 20%用于测试
        train_dataset, test_dataset = torch.utils.data.random_split(combined_dataset, [train_size, test_size])

        # 创建训练集和测试集的数据加载器
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        # 创建生成数据集的数据加载器
        generated_dataset = TensorDataset(generated_X, generated_Y)
        generated_loader = DataLoader(generated_dataset, batch_size=128, shuffle=True)

        # 训练模型
        print(f"Starting model training for client {self.client_id}...")
        for epoch in tqdm(range(self.epochs)):
            self.model.train_model_old(train_loader=train_loader, test_loader=test_loader, generated_loader=None,
                                       epoch=epoch)

        # 微调模型
        print(f"Starting model fine-tuning for client {self.client_id}...")
        for epoch in tqdm(range(self.epochs)):
            self.model.fine_tune_old(train_loader=train_loader, test_loader=test_loader, generated_loader=None, epoch=epoch)

        # 保存模型
        print(f"Saving model for client {self.client_id}...")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def evaluate(self, test_loader):
        # 评估模型在测试集上的性能
        self.model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features, mode="client")
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        print(f'Client{self.client_id}, Test Accuracy: {accuracy:.4f}')

    def get_CRNet_model(self):
        return self.CRNet

    def train_self_CRNet(self, load=True):
        # 创建模型保存目录
        model_dir = os.path.join('models', 'CRNet_dict')
        os.makedirs(model_dir, exist_ok=True)

        # 根据 client_id 生成模型文件名
        model_filename = f"client_{self.client_id}_CRNet.pth"
        model_path = os.path.join(model_dir, model_filename)

        # 检查是否存在保存的模型
        # if os.path.exists(model_path) and load:
        #     print(f"Loading existing CRNet model for client {self.client_id}...")
        #     self.CRNet.load_state_dict(torch.load(model_path))
        #     print(f"Client {self.client_id} CRNet model loaded successfully.")
        #     return True
        # else:
        #     print(f"No existing CRNet model found for client {self.client_id}. Starting with a new model.")

        combined_X = torch.cat((self.unaligned_dataset['X'], self.aligned_dataset['X']), dim=0)
        combined_Y = torch.cat((self.unaligned_dataset['Y'], self.aligned_dataset['Y']), dim=0)
        generated_X = self.generated_dataset['X']
        generated_Y = self.generated_dataset['Y']

        with torch.no_grad():
            combined_features = self.model.base_model(combined_X.to(self.device), mode='fet').to(self.device)

        # 将数据集划分为训练集和验证集
        train_features, val_features, train_labels, val_labels = train_test_split(
            combined_features.cpu().numpy(),
            combined_Y.cpu().numpy(),
            test_size=0.1,  # 10% 用于验证
            random_state=42
        )

        # 转换为张量
        train_features = torch.from_numpy(train_features).to(self.device)
        val_features = torch.from_numpy(val_features).to(self.device)
        train_labels = torch.from_numpy(train_labels).to(self.device)
        val_labels = torch.from_numpy(val_labels).to(self.device)

        idx_to_label = {
            0: "very uncomfortable",
            1: "uncomfortable",
            2: "slightly uncomfortable",
            3: "slightly comfortable",
            4: "comfortable",
            5: "very comfortable"
        }

        print(f"Starting CRNet training for client {self.client_id}...")
        train_and_eval_CRNet(self.CRNet, train_features, train_labels, val_features, val_labels,
                             idx_to_label=idx_to_label)

        # 保存模型
        print(f"Saving CRNet model for client {self.client_id}...")
        torch.save(self.CRNet.state_dict(), model_path)
        print(f"CRNet model saved to {model_path}")

    def train_self_CRNet_old(self):
        combined_X = torch.cat((self.unaligned_dataset['X'], self.aligned_dataset['X']), dim=0)
        combined_Y = torch.cat((self.unaligned_dataset['Y'], self.aligned_dataset['Y']), dim=0)
        test_X = self.test_dataset['X']
        test_Y = self.test_dataset['Y']
        with torch.no_grad():
            combined_features = self.model.base_model(combined_X.to(self.device), mode='fet').to(
                self.device)
            test_features = self.model.base_model(test_X.to(self.device), mode='fet').to(
                self.device)
        # 将数据集划分为训练集和验证集
        train_features, val_features, train_labels, val_labels = train_test_split(
            combined_features.cpu().numpy(),
            combined_Y.cpu().numpy(),
            test_size=0.1,  # 10% 用于验证
            random_state=42
        )
        # 转换为张量
        train_features = torch.from_numpy(train_features).to(self.device)
        val_features = test_features.to(self.device) # torch.from_numpy(val_features).to(self.device)
        train_labels = torch.from_numpy(train_labels).to(self.device)
        val_labels = combined_Y.to(self.device) # torch.from_numpy(val_labels).to(self.device)
        idx_to_label = {
            0: "very uncomfortable",
            1: "uncomfortable",
            2: "slightly uncomfortable",
            3: "slightly comfortable",
            4: "comfortable",
            5: "very comfortable"
        }
        train_and_eval_CRNet(self.CRNet, train_features, train_labels, val_features, val_labels, idx_to_label=idx_to_label)

        # test_CRNet(self.CRNet, val_features, val_labels, idx_to_label=idx_to_label)

    def train_base_model_aligned(self, load=True):
        """
        只使用aligned数据训练base_model (MLP3)

        参数:
        - load: bool, 是否加载已存在的模型
        """
        # 创建模型保存目录
        model_dir = os.path.join('models', 'aligned_base_model')
        os.makedirs(model_dir, exist_ok=True)

        # 根据 client_id 生成模型文件名
        model_filename = f"client_{self.client_id}_aligned_base.pth"
        model_path = os.path.join(model_dir, model_filename)

        # 初始化日志
        log_dir = os.path.join('logs', 'aligned_base_model')
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"client_{self.client_id}_training.log")
        logging.basicConfig(filename=log_filename, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"Starting training for client {self.client_id}...")

        # # 检查是否存在保存的模型
        # if os.path.exists(model_path) and load:
        #     logging.info(f"Loading existing aligned base model for client {self.client_id}...")
        #     self.model.base_model.load_state_dict(torch.load(model_path))
        #     logging.info("Aligned base model loaded successfully.")
        #     return True

        # 准备数据
        combined_X = torch.cat((self.unaligned_dataset['X'], self.aligned_dataset['X']), dim=0)
        combined_Y = torch.cat((self.unaligned_dataset['Y'], self.aligned_dataset['Y']), dim=0)

        train_dataset = TensorDataset(combined_X, combined_Y)
        test_X = self.test_dataset['X']
        test_Y = self.test_dataset['Y']
        val_dataset = TensorDataset(test_X, test_Y)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.base_model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        # 训练参数
        num_epochs = 100
        best_val_acc = 0
        patience = 10
        patience_counter = 0

        # 训练循环
        logging.info(f"Starting aligned base model training for client {self.client_id}...")
        for epoch in range(num_epochs):
            # 训练模式
            self.base_model.train()
            total_loss = 0
            correct = 0
            total = 0

            # 每个类的正确预测统计
            class_correct = defaultdict(int)
            class_total = defaultdict(int)

            # 训练阶段
            for batch_X, batch_Y in train_loader:
                batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)

                optimizer.zero_grad()
                outputs = self.base_model(batch_X, mode='SixCls')
                loss = criterion(outputs, batch_Y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_Y.size(0)
                correct += (predicted == batch_Y).sum().item()

                # 统计每个类的准确率
                for label, prediction in zip(batch_Y, predicted):
                    class_total[label.item()] += 1
                    if label == prediction:
                        class_correct[label.item()] += 1

            train_acc = 100 * correct / total
            avg_loss = total_loss / len(train_loader)

            # 每个类的准确率
            class_accuracies = {cls: 100 * class_correct[cls] / class_total[cls]
            if class_total[cls] > 0 else 0.0
                                for cls in class_total.keys()}

            # 验证阶段
            # 验证阶段
            self.model.base_model.eval()
            val_correct = 0
            val_total = 0

            # 每个类的验证正确统计
            val_class_correct = defaultdict(int)
            val_class_total = defaultdict(int)

            with torch.no_grad():
                for batch_X, batch_Y in val_loader:
                    batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
                    outputs = self.base_model(batch_X, mode='SixCls')
                    _, predicted = torch.max(outputs.data, 1)

                    val_total += batch_Y.size(0)
                    val_correct += (predicted == batch_Y).sum().item()

                    # 统计每个类的验证准确率
                    for label, prediction in zip(batch_Y, predicted):
                        val_class_total[label.item()] += 1
                        if label == prediction:
                            val_class_correct[label.item()] += 1

            val_acc = 100 * val_correct / val_total

            # 每个类的验证准确率
            val_class_accuracies = {cls: 100 * val_class_correct[cls] / val_class_total[cls]
            if val_class_total[cls] > 0 else 0.0
                                    for cls in val_class_total.keys()}

            # 记录日志
            logging.info(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f} '
                         f'Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}% '
                         f'Per-class Val Accuracies: {val_class_accuracies}')

            # 打印训练进度
            print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f} '
                  f'Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}% '
                  f'Per-class Val Accuracies: {val_class_accuracies}')

            # 记录日志
            # logging.info(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f} '
            #              f'Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%'
            #              f'Per-class Train Accuracies: {class_accuracies}')
            # # logging.info(f'Per-class Train Accuracies: {class_accuracies}')
            #
            # # 打印训练进度
            # print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f} '
            #       f'Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%'
            #       f'Per-class Train Accuracies: {class_accuracies}')

            # 早停策略
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.base_model.state_dict(), model_path)
                logging.info(f"Saved best model with validation accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        # 加载最佳模型
        self.base_model.load_state_dict(torch.load(model_path))
        logging.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}")
        print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}")

    def extract_feature(self):
        """
        提取对齐数据集的特征
        :return: 特征张量
        """
        with torch.no_grad():
            aligned_features = self.model.base_model(self.aligned_dataset['X'].to(self.device), mode='fet').to(
                self.device)
            return aligned_features


    def extract_feature_gen(self):
        with torch.no_grad():
            # client_data = {
            #     'features': np.random.rand(60, 16), # generated_features
            #     'labels': np.repeat(np.arange(6), 10)
            # }
            generated_features = self.model.base_model(self.generated_dataset['X'].to(self.device),
                                                       mode='fet').detach().cpu().numpy()
            generated_labels = self.generated_dataset['Y'].detach().cpu().numpy()
            # print(np.unique(generated_labels))
            # print(generated_features.shape)
            client_data = {
                'features': generated_features,
                'labels': generated_labels
            }
            # print(client_data)
            return client_data

    def extract_feature_aligned(self):
        with torch.no_grad():
            generated_features = self.model.base_model(self.aligned_dataset['X'].to(self.device),
                                                       mode='fet').detach().cpu().numpy()
            generated_labels = self.aligned_dataset['Y'].detach().cpu().numpy()
            client_data = {
                'features': generated_features,
                'labels': generated_labels
            }
            return client_data

    def extract_feature_unaligned(self):
        with torch.no_grad():
            generated_features = self.model.base_model(self.unaligned_dataset['X'].to(self.device),
                                                       mode='fet').detach().cpu().numpy()
            generated_labels = self.unaligned_dataset['Y'].detach().cpu().numpy()
            client_data = {
                'features': generated_features,
                'labels': generated_labels
            }
            return client_data


    def extract_feature_test(self, test_dataset):
        with torch.no_grad():
            test_features = self.model.base_model(test_dataset['X'].to(self.device), mode='fet').to(self.device)
            return test_features


    def update_local_model(self, local_model, global_model, update_ratio=0.5):
        """
        更新本地模型的参数。

        参数:
        - local_model: nn.Module，本地模型。
        - global_model: nn.Module，全局模型。
        - update_ratio: float，更新比例，1.0 表示完全替换，0.0 表示不更新。

        返回:
        - 更新后的本地模型。
        """

        # 获取全局模型的参数
        global_state_dict = global_model.state_dict()

        # 更新本地模型的参数
        local_state_dict = local_model.state_dict()

        for name, global_param in global_state_dict.items():
            if name in local_state_dict:
                local_param = local_state_dict[name]
                # 更新参数: (1 - update_ratio) * local_param + update_ratio * global_param
                updated_param = (1 - update_ratio) * local_param + update_ratio * global_param
                local_state_dict[name].copy_(updated_param)

        # 加载更新后的参数到本地模型
        local_model.load_state_dict(local_state_dict)

        return local_model

    def generate_pseudo_labels_for_tabular(self):
        """
        为表格数据生成伪标签并创建新的数据集

        Returns:
            dict: 包含带伪标签的训练集和测试集
        """

        import numpy as np

        # 合并未对齐和对齐数据集
        combined_X = torch.cat((self.unaligned_dataset['X'], self.aligned_dataset['X']), dim=0)

        # 将数据转换为numpy数组并标准化
        combined_data = combined_X.cpu().numpy()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(combined_data)

        # 使用K-means进行聚类
        kmeans = KMeans(
            n_clusters=self.cluster_num,
            random_state=42,
            n_init=10  # 多次初始化以获得更好的聚类结果
        )
        pseudo_labels = kmeans.fit_predict(scaled_data)
        pseudo_labels = torch.tensor(pseudo_labels)

        # 计算每个样本到其聚类中心的距离（用于置信度计算）
        distances = kmeans.transform(scaled_data)  # 返回每个样本到所有聚类中心的距离
        min_distances = distances.min(axis=1)  # 每个样本到其聚类中心的距离
        confidence_scores = 1 / (1 + min_distances)  # 转换为置信度分数
        confidence_scores = (confidence_scores - confidence_scores.min()) / (
                    confidence_scores.max() - confidence_scores.min())

        self.pseudo_label = pseudo_labels
        # 获取数据集大小并随机打乱
        # dataset_size = len(combined_X)
        # indices = torch.randperm(dataset_size)
        # train_size = int(0.9 * dataset_size)
        #
        # train_indices = indices[:train_size]
        # test_indices = indices[train_size:]
        #
        # # 创建训练集
        # train_X = combined_X[train_indices]
        # train_Y = pseudo_labels[train_indices]
        # train_confidence = torch.tensor(confidence_scores[train_indices.numpy()])
        #
        # # 创建测试集
        # test_X = combined_X[test_indices]
        # test_Y = pseudo_labels[test_indices]
        # test_confidence = torch.tensor(confidence_scores[test_indices.numpy()])

        # 创建新的数据集字典
        # self.pseudo_labeled_tabular = {
        #     'train': {
        #         'X': train_X,
        #         'Y': train_Y,
        #         'confidence': train_confidence
        #     },
        #     'test': {
        #         'X': test_X,
        #         'Y': test_Y,
        #         'confidence': test_confidence
        #     },
        #     'cluster_info': {
        #         'centers': kmeans.cluster_centers_,
        #         'scaler': scaler,
        #         'cluster_sizes': np.bincount(pseudo_labels.numpy())
        #     }
        # }
        #
        # # 打印聚类信息
        # print("\nClustering Results:")
        # print(f"Number of clusters: {self.cluster_num}")
        # for i in range(self.cluster_num):
        #     cluster_size = (pseudo_labels == i).sum().item()
        #     print(f"Cluster {i}: {cluster_size} samples ({cluster_size / len(pseudo_labels) * 100:.2f}%)")
        # print(f"Average confidence score: {confidence_scores.mean():.3f}")

        return self.pseudo_label # self.pseudo_labeled_tabular