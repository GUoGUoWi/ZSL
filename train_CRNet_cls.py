import datetime
import logging
import os


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix
from etc.utils import generate_label_matrix
from models.CRNet.trainer import generate_label_matrix_clip

THERMAL_COMFORT_DESCRIPTIONS_SEEN = {
    0: "very cold uncomfortable thermal condition",
    1: "cold slightly uncomfortable thermal environment",
    2: "slightly cool but acceptable thermal state",
    3: "neutral comfortable thermal condition",
    4: "slightly warm but acceptable thermal state",
    5: "hot uncomfortable thermal environment"
}
MODELNET_DESCRIPTIONS_SEEN = {
    0:'bathtub', 1:'bed', 2:'chair', 3:'desk', 4:'dresser', 5:'monitor', 6:'night_stand', 7:'sofa', 8:'table', 9:'toilet'
}

FMNIST_DESCRIPTIONS_SEEN = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

CIFAR10_DESCRIPTIONS_DETAILED = {
    0: 'an airplane, a fixed-wing aircraft',
    1: 'an automobile, a passenger car',
    2: 'a bird, a small flying animal',
    3: 'a cat, a domestic feline pet',
    4: 'a deer, a wild forest animal with antlers',
    5: 'a dog, a domestic canine pet',
    6: 'a frog, an amphibian with smooth skin',
    7: 'a horse, a large four-legged mammal',
    8: 'a ship, a large watercraft vessel',
    9: 'a truck, a large road vehicle for carrying cargo'
}


def convert_label_indices_to_vectors(label_indices, idx_to_label, glove_model_name="./data/glove-wiki-gigaword-200.txt"):
    labels = [idx_to_label[idx.item()] for idx in label_indices]
    label_matrix = generate_label_matrix(labels, glove_model_name)
    return label_matrix

def contrastive_loss(semantic_similarity, labels):
    def create_contrastive_labels(labels):
        num_classes = 6
        labels_onehot = F.one_hot(labels, num_classes=num_classes).float()
        similarity_target = torch.matmul(labels_onehot, labels_onehot.t())
        return similarity_target
    target_similarity = create_contrastive_labels(labels)
    # 使用InfoNCE loss
    temperature = 0.1
    semantic_similarity = semantic_similarity / temperature
    # 计算对比损失
    exp_sim = torch.exp(semantic_similarity)
    log_prob = semantic_similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
    mean_log_prob = (target_similarity * log_prob).sum(dim=1).mean()

    return -mean_log_prob


def train_and_eval_CRNet(model, train_visual_features, train_label_indices, val_visual_features, val_label_indices,
                         idx_to_label, num_epochs=40, learning_rate=0.0005, device='cuda'):
    model.to(device)
    train_visual_features = train_visual_features.to(torch.float32)
    train_semantic_vectors = model.get_semantic()
    # 生成正负样本对
    train_dataset = TensorDataset(train_visual_features, train_label_indices)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_visual_features = val_visual_features.to(device)
    test_dataset = TensorDataset(val_visual_features, val_label_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    ce_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tree_loss = 0.0
        epoch_contrast_loss = 0.0
        for visual_features, labels in train_loader:
            visual_features = visual_features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            probabilities, semantic_similarity = model(visual_features)
            loss_ce = ce_criterion(probabilities, labels)
            contrast_loss = contrastive_loss(semantic_similarity, labels)
            # 计算树结构感知损失
            all_levels = torch.arange(6, device=labels.device).float()
            labels_expanded = labels.unsqueeze(1).float()
            level_distances = torch.abs(labels_expanded - all_levels) / (6 - 1)
            expected_scores = 1.0 - level_distances
            tree_loss = torch.mean((F.softmax(probabilities, dim=-1) - expected_scores) ** 2)
            total_loss = 1.0 * loss_ce + 1.0 * contrast_loss
            loss = total_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_tree_loss += tree_loss.item()
            epoch_contrast_loss += contrast_loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], CRNet Train Loss: {epoch_loss / len(train_loader):.4f},'
              f'Contrast_loss: {epoch_contrast_loss / len(train_loader):.4f},'
              f'Tree Loss: {epoch_tree_loss / len(train_loader):.4f}')
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], CRNet Train Loss: {epoch_loss / len(train_loader):.4f},'
                     f'Contrast_loss: {epoch_contrast_loss / len(train_loader):.4f},'
                     f'Tree Loss: {epoch_tree_loss / len(train_loader):.4f}')

        model.eval()  # 设置模型为评估模式
        # 初始化每个类别的统计数据
        class_correct = {i: 0 for i in range(6)}  # 假设有6个类别
        class_total = {i: 0 for i in range(6)}

        predictions = []
        true_labels = []

        with torch.no_grad():
            for visual_features, label in test_loader:
                visual_features = visual_features.to(device)
                label = label.to(device)
                similarity_scores, semantic_similarity = model(visual_features)
                similarity_scores_softmax = F.softmax(similarity_scores, dim=1)
                _, predicted_indices = torch.max(similarity_scores_softmax, dim=1)

                # 更新每个类别的统计数据
                for pred, true_label in zip(predicted_indices, label):
                    class_total[true_label.item()] += 1
                    if pred == true_label:
                        class_correct[true_label.item()] += 1

                true_labels.extend(label.cpu().numpy())
                predictions.extend(predicted_indices.cpu().numpy())

        # 计算总体准确率
        total_correct = sum(class_correct.values())
        total_samples = sum(class_total.values())
        overall_accuracy = total_correct / total_samples

        # 打印总体准确率
        print(f'Epoch [{epoch + 1}/{num_epochs}], Overall Test Accuracy: {overall_accuracy:.4f}')
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Overall Test Accuracy: {overall_accuracy:.4f}')

        # 打印每个类别的准确率
        print("\nPer-class Accuracy:")
        logging.info("\nPer-class Accuracy:")
        for class_idx in range(6):
            if class_total[class_idx] > 0:  # 避免除以零
                class_accuracy = class_correct[class_idx] / class_total[class_idx]
                class_name = idx_to_label[class_idx] if idx_to_label else f"Class {class_idx}"
                print(f"{class_name}: {class_accuracy:.4f} ({class_correct[class_idx]}/{class_total[class_idx]})")
                logging.info(
                    f"{class_name}: {class_accuracy:.4f} ({class_correct[class_idx]}/{class_total[class_idx]})")

    return model


# 训练函数
def train_and_eval_CRNet_old(model, train_visual_features, train_label_indices, val_visual_features, val_label_indices, idx_to_label, num_epochs=40, learning_rate=0.0005, device='cuda'):
    model.to(device)
    train_visual_features = train_visual_features.to(torch.float32)
    train_semantic_vectors = model.get_semantic()
    # 生成正负样本对
    train_dataset = TensorDataset(train_visual_features, train_label_indices)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_visual_features = val_visual_features.to(device)
    test_dataset = TensorDataset(val_visual_features, val_label_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    ce_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tree_loss = 0.0
        epoch_contrast_loss = 0.0
        for visual_features, labels in train_loader:
            visual_features = visual_features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            probabilities, semantic_similarity = model(visual_features)
            loss_ce = ce_criterion(probabilities, labels)
            contrast_loss = contrastive_loss(semantic_similarity, labels)
            # 计算树结构感知损失
            # 创建所有可能的级别 (0-5)
            all_levels = torch.arange(6, device=labels.device).float()
            # 将labels扩展为 [batch_size, 1]
            labels_expanded = labels.unsqueeze(1).float()
            # 计算每个样本对所有级别的距离矩阵 [batch_size, num_classes]
            level_distances = torch.abs(labels_expanded - all_levels) / (6 - 1)
            # 计算期望的相似度分数
            expected_scores = 1.0 - level_distances
            # 计算预测概率与期望分数之间的均方差损失
            tree_loss = torch.mean((F.softmax(probabilities, dim=-1) - expected_scores) ** 2)
            total_loss = 1.0 * loss_ce + 1.0 * contrast_loss
            loss = total_loss
            # loss = loss_ce
            # loss = contrast_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_tree_loss += tree_loss.item()
            epoch_contrast_loss += contrast_loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], CRNet Train Loss: {epoch_loss / len(train_loader):.4f},'
              f'Contrast_loss: {epoch_contrast_loss / len(train_loader):.4f},'
              f'Tree Loss: {epoch_tree_loss / len(train_loader):.4f}')
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], CRNet Train Loss: {epoch_loss / len(train_loader):.4f},'
              f'Contrast_loss: {epoch_contrast_loss / len(train_loader):.4f},'
              f'Tree Loss: {epoch_tree_loss / len(train_loader):.4f}')
        # print(f'Epoch [{epoch + 1}/{num_epochs}], CRNet Train Loss: {epoch_loss / len(train_loader):.4f}')

        model.eval()  # 设置模型为评估模式
        cls_total_correct = 0
        sem_total_correct = 0
        cls_total = 0
        predictions = []
        predictions_sem = []
        true_labels = []
        for visual_features, label in test_loader:
            visual_features = visual_features.to(device)
            label = label.to(device)
            similarity_scores, semantic_similarity = model(visual_features)
            similarity_scores_softmax = F.softmax(similarity_scores, dim=1)
            _, predicted_indices = torch.max(similarity_scores_softmax, dim=1)
            # 计算基于semantic similarity的预测
            # semantic_similarity_sigmoid = F.sigmoid(semantic_similarity)
            # _, predicted_sem = torch.max(semantic_similarity_sigmoid, dim=1)
            cls_total_correct += (predicted_indices == label).sum().item()
            # sem_total_correct += (predicted_sem == label).sum().item()
            true_labels += label.tolist()
            predictions += predicted_indices.tolist()
            # predictions_sem += predicted_sem.tolist()

        # print('True_labels:', true_labels)
        # print('cls_predictions:', predictions)
        # print('sem_predictions:', predictions_sem)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Classify Test Accuracy: {cls_total_correct/ len(predictions):.4f}')
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Classify Test Accuracy: {cls_total_correct / len(predictions):.4f}')
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Similarity Test Accuracy: {sem_total_correct/ len(predictions):.4f}')


def calculate_mutual_information(x, y, n_bins=50):
    """
    计算两个特征之间的互信息
    x, y: torch.Tensor, 特征向量
    """
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    # 将特征转换为离散值
    x_bins = np.histogram_bin_edges(x, bins=n_bins)
    y_bins = np.histogram_bin_edges(y, bins=n_bins)

    x_discrete = np.digitize(x, x_bins)
    y_discrete = np.digitize(y, y_bins)

    # 计算联合概率和边缘概率
    xy_hist = np.histogram2d(x_discrete, y_discrete, bins=[n_bins, n_bins])[0]
    p_xy = xy_hist / xy_hist.sum()
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    # 计算互信息
    mi = 0
    for i in range(n_bins):
        for j in range(n_bins):
            if p_xy[i, j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))

    return mi

def test_CRNet(models, client_cluster_centers, visual_features, labels, weights, device='cuda', dataset='modelnet', clip=False):
    """
    对CRNet模型进行零-shot测试，并计算准确率。
    tuple: (预测标签, 准确率)
    """
    if dataset == 'comfort':
        DESCRIPTIONS = THERMAL_COMFORT_DESCRIPTIONS_SEEN
        num_label = 6
    elif dataset == 'modelnet':
        DESCRIPTIONS = MODELNET_DESCRIPTIONS_SEEN
        num_label = 10
    elif dataset == 'fmnist':
        if clip:
            DESCRIPTIONS = CIFAR10_DESCRIPTIONS_DETAILED
        else:
            DESCRIPTIONS = FMNIST_DESCRIPTIONS_SEEN
        num_label = 10
    else:
        raise RuntimeError('No such dataset')
    descriptions = [DESCRIPTIONS[i] for i in list(range(num_label))]
    if clip:
        semantic_prototypes = generate_label_matrix_clip(descriptions)
    else:
        semantic_prototypes = generate_label_matrix(descriptions,
                                                              glove_model_name="./data/glove-wiki-gigaword-200.txt")
    # semantic_prototypes = generate_label_matrix(descriptions, glove_model_name="./data/glove-wiki-gigaword-200.txt")
    class_prototypes = torch.from_numpy(semantic_prototypes).float().to(device)
    total = 0
    total_correct = 0
    num_classes = num_label
    predictions = []
    num_clients = len(models)
    # client_class_weights = {
    #     0: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.2},  # 客户端0对1,2,3,4类较可信
    #     1: {0: 1.0, 1: 1.0, 2: 0.2, 3: 0.2, 4: 1.0, 5: 1.0},  # 客户端1对2,3,4,5类较可信
    #     2: {0: 1.0, 1: 1.0, 2: 0.2, 3: 1.0, 4: 1.0, 5: 1.0}  # 客户端2对1,3,4,5类较可信
    # }
    num_samples = visual_features[0].shape[0]
    for i in range(len(visual_features)):
        print('num_test_samples', visual_features[i].shape)
        print('num_labels', labels.shape)
    final_predictions = np.zeros(num_samples, dtype=int)
    # 存储每个模型的预测结果和概率
    all_predictions = []
    all_predictions_sem = []
    all_probabilities = []
    all_probabilities_sem = []
    all_features = []
    individual_accuracies = []  # 存储每个客户端的准确率
    # 对每个客户端进行预测
    # 将labels转换为numpy数组，方便计算
    if isinstance(labels, torch.Tensor):
        true_labels = labels.cpu().numpy()
    else:
        true_labels = labels

    for client_id in range(len(models)):
        model = models[client_id].eval()
        client_cluster_center = client_cluster_centers[client_id]
        features = torch.from_numpy(visual_features[client_id]).to(torch.float32).to(device)
        with torch.no_grad():
            outputs, visual_common, semantic_common = model(
                features,
                class_prototypes,
                client_cluster_center,
                return_embeddings=True)
            probabilities = outputs
            # print('output', outputs)
            print('output', outputs.shape)
            predictions = torch.argmax(probabilities, dim=1)
        # 添加新的后处理逻辑处理
        # for i in range(len(predictions[:100])):
        #     max_value, max_index = torch.max(probabilities[i], dim=-1)
        #     if max_value < 0.35:
        #         predictions[i] = 0
        #         probabilities[i][0] = 1.0

        client_preds = predictions.cpu().numpy()
        all_predictions.append(predictions.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())
        all_features.append((semantic_common, visual_common))
        # 计算每个客户端的准确率
        client_accuracy = accuracy_score(true_labels, client_preds)
        client_accuracy_with_unseen = accuracy_score(true_labels[:100], client_preds[:100])
        individual_accuracies.append(client_accuracy)
        print(f"Client {client_id} Accuracy: {client_accuracy:.4f}")
        print(f"Client {client_id} Accuracy with unseen: {client_accuracy_with_unseen:.4f}")
        logging.info(f"Client {client_id} Accuracy: {client_accuracy:.4f}")
        logging.info(f"Client {client_id} Accuracy with unseen: {client_accuracy_with_unseen:.4f}")

    # 计算客户端之间的互信息矩阵
    mi_matrix = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(num_clients):
            if i != j:
                # 计算feature_anchors之间的互信息
                mi_anchors = calculate_mutual_information(
                    all_features[i][0].mean(dim=0),
                    all_features[j][0].mean(dim=0)
                )
                # 计算visual_features之间的互信息
                mi_visual = calculate_mutual_information(
                    all_features[i][1].mean(dim=0),
                    all_features[j][1].mean(dim=0)
                )
                mi_matrix[i, j] = (mi_anchors + mi_visual) / 2


    # 使用权重和互信息进行集成预测
    final_predictions = np.zeros(num_samples, dtype=int)
    for sample_idx in range(num_samples):
        class_votes = np.zeros(num_classes)
        # 对每对客户端的预测进行加权
        for i in range(num_clients):
            for j in range(num_clients):
                if i != j and i < j:
                    # 使用客户端权重和互信息权重计算最终预测
                    # print(all_probabilities[i][sample_idx].shape)
                    # print(weights[i].detach().cpu().numpy().shape)
                    pair_pred = (
                            all_probabilities[i][sample_idx] * weights[i].detach().cpu().numpy() +
                            all_probabilities[j][sample_idx] * weights[j].detach().cpu().numpy()
                    )
                    class_votes += mi_matrix[i, j] * pair_pred

        final_predictions[sample_idx] = np.argmax(class_votes)

    # 根据互信息计算每个客户端的权重
    # client_weights = mi_matrix.sum(axis=1)
    # client_weights = client_weights / client_weights.sum()
    # # 加权投票预测
    # final_predictions = np.zeros(num_samples, dtype=int)
    # for sample_idx in range(num_samples):
    #     class_votes = np.zeros(num_classes)
    #     for client_id in range(num_clients):
    #         pred_class = all_predictions[client_id][sample_idx]
    #         prob = all_probabilities[client_id][sample_idx][pred_class]
    #         class_votes[pred_class] += prob * client_weights[client_id]
    #
    #     final_predictions[sample_idx] = np.argmax(class_votes)

    # 计算准确率
    if isinstance(labels, torch.Tensor):
        true_labels = labels.cpu().numpy()
    else:
        true_labels = labels
    ensemble_accuracy = accuracy_score(true_labels, final_predictions)
    print('1',true_labels, final_predictions)
    ensemble_accuracy_without_seen = accuracy_score(true_labels[:100], final_predictions[:100])

    print(f"\nEnsemble Model Accuracy: {ensemble_accuracy:.4f}")
    print(f"\nEnsemble Model Accuracy without seen: {ensemble_accuracy_without_seen:.4f}")
    logging.info(f"\nEnsemble Model Accuracy: {ensemble_accuracy:.4f}")
    logging.info(f"\nEnsemble Model Accuracy without seen: {ensemble_accuracy_without_seen:.4f}")

    # 打印混淆矩阵
    conf_matrix = confusion_matrix(true_labels, final_predictions)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # 计算每个类别的准确率
    class_accuracies = {}
    for class_idx in range(num_classes):
        class_mask = (true_labels == class_idx)
        if np.any(class_mask):  # 确保有该类别的样本
            class_acc = accuracy_score(true_labels[class_mask], final_predictions[class_mask])
            class_accuracies[class_idx] = class_acc
            print(f"Class {class_idx} Accuracy: {class_acc:.4f}")
            logging.info(f"Class {class_idx} Accuracy: {class_acc:.4f}")


def train_and_eval_MLP(model, train_features, train_labels, val_features, val_labels,
                       idx_to_label, num_epochs=20, learning_rate=1e-3,
                       batch_size=128, device='cuda'):
    """
    训练和评估普通 MLP 模型，记录日志并返回每个类别的准确率。

    参数:
    - model: MLP 模型
    - train_features: 训练集特征
    - train_labels: 训练集标签
    - val_features: 验证集特征
    - val_labels: 验证集标签
    - idx_to_label: 类别索引到类别标签的映射
    - num_epochs: 训练轮数
    - learning_rate: 学习率
    - batch_size: 批次大小
    - device: 训练设备 ('cuda' 或 'cpu')

    返回:
    - class_accuracies: 每个类别的准确率字典
    - history: 训练历史记录，包含损失和准确率
    """
    # 初始化日志
    log_dir = os.path.join('logs', 'MLP_training')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f'training_{timestamp}.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting MLP training...")

    # 记录训练历史
    history = {
        'train_loss': [],
        'val_accuracy': [],
        'class_accuracies': []
    }

    # 模型和数据准备
    model.to(device)
    train_features = train_features.to(torch.float32)
    train_labels = train_labels.to(torch.long)

    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_features = val_features.to(torch.float32).to(device)
    val_labels = val_labels.to(torch.long)
    val_dataset = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0
    best_model_state = None

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features, mode='SixCls')
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}')

        # 验证阶段
        model.eval()
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_features, mode='SixCls')
                _, predicted = torch.max(outputs, 1)

                total_correct += (predicted == batch_labels).sum().item()
                total_samples += batch_labels.size(0)

                # 统计每个类别的准确率
                for label, pred in zip(batch_labels, predicted):
                    label_item = label.item()
                    class_total[label_item] += 1
                    if label_item == pred.item():
                        class_correct[label_item] += 1

        # 计算总体准确率
        overall_accuracy = total_correct / total_samples
        history['val_accuracy'].append(overall_accuracy)

        # 计算并记录每个类别的准确率
        class_accuracies = {idx_to_label[cls]: 100 * class_correct[cls] / class_total[cls]
        if class_total[cls] > 0 else 0.0
                            for cls in class_total.keys()}
        history['class_accuracies'].append(class_accuracies)

        # 记录日志
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], '
                     f'Overall Validation Accuracy: {overall_accuracy:.4f}')
        logging.info(f'Class-wise Accuracies: {class_accuracies}')

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Overall Validation Accuracy: {overall_accuracy:.4f}')
        print(f'Class-wise Accuracies: {class_accuracies}')

        # 保存最佳模型
        if overall_accuracy > best_accuracy:
            best_accuracy = overall_accuracy
            best_model_state = model.state_dict().copy()
            logging.info(f'New best model saved with accuracy: {best_accuracy:.4f}')

    # 训练结束，加载最佳模型
    model.load_state_dict(best_model_state)
    logging.info(f"Training completed. Best validation accuracy: {best_accuracy:.4f}")
    print(f"Training completed. Best validation accuracy: {best_accuracy:.4f}")

    return class_accuracies, history


def train_and_eval_ResNet(model, train_loader, val_loader, idx_to_label,
                          num_epochs=20, learning_rate=1e-3, device='cuda'):
    """
    训练和评估 ResNet 模型，记录日志并返回每个类别的准确率。

    参数:
    - model: ResNet 模型
    - train_loader: 训练数据加载器
    - val_loader: 验证数据加载器
    - idx_to_label: 类别索引到类别标签的映射
    - num_epochs: 训练轮数
    - learning_rate: 学习率
    - device: 训练设备 ('cuda' 或 'cpu')

    返回:
    - class_accuracies: 每个类别的准确率字典
    - history: 训练历史记录，包含损失和准确率
    """
    # 初始化日志
    log_dir = os.path.join('logs', 'ResNet_training')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f'training_{timestamp}.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting ResNet training...")

    # 记录训练历史
    history = {
        'train_loss': [],
        'val_accuracy': [],
        'class_accuracies': []
    }

    # 模型准备
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_accuracy = 0.0
    best_model_state = None

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

        avg_loss = epoch_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        history['train_loss'].append(avg_loss)

        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], '
                     f'Train Loss: {avg_loss:.4f}, '
                     f'Train Accuracy: {train_accuracy:.2f}%')
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {avg_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%')

        # 验证阶段
        model.eval()
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_images)
                _, predicted = torch.max(outputs, 1)

                total_correct += (predicted == batch_labels).sum().item()
                total_samples += batch_labels.size(0)

                # 统计每个类别的准确率
                for label, pred in zip(batch_labels, predicted):
                    label_item = label.item()
                    class_total[label_item] += 1
                    if label_item == pred.item():
                        class_correct[label_item] += 1

        # 计算总体准确率
        overall_accuracy = 100 * total_correct / total_samples
        history['val_accuracy'].append(overall_accuracy)

        # 计算并记录每个类别的准确率
        class_accuracies = {idx_to_label[cls]: 100 * class_correct[cls] / class_total[cls]
        if class_total[cls] > 0 else 0.0
                            for cls in class_total.keys()}
        history['class_accuracies'].append(class_accuracies)

        # 记录日志
        logging.info(f'Validation Accuracy: {overall_accuracy:.2f}%')
        logging.info(f'Class-wise Accuracies: {class_accuracies}')

        print(f'Validation Accuracy: {overall_accuracy:.2f}%')
        print(f'Class-wise Accuracies: {class_accuracies}')

        # 保存最佳模型
        if overall_accuracy > best_accuracy:
            best_accuracy = overall_accuracy
            best_model_state = model.state_dict().copy()

            # 保存检查点
            checkpoint_path = os.path.join(log_dir, f'best_model_{timestamp}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'class_accuracies': class_accuracies
            }, checkpoint_path)

            logging.info(f'Saved best model with accuracy: {best_accuracy:.2f}%')
            print(f'Saved best model with accuracy: {best_accuracy:.2f}%')

        # 更新学习率
        scheduler.step()

    # 恢复最佳模型状态
    model.load_state_dict(best_model_state)

    return class_accuracies, history
