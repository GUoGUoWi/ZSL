import copy
import logging
import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from client.client_template_baseline import VFLClient
from client.ensemble_learning_server import EnsemblePredictor

from data.comfort_dataset import ComfortDataset
from data.modelnet_dataset import MultiViewDataset
from etc.prepare_exp import get_datasets_tab
from etc.utils import setup_logging, set_random_seed, calculate_tensors_size_detailed
from models.CRNet.models_cls import CRnetCls
from models.CRNet.train_CRNet_cls import train_and_eval_CRNet, train_and_eval_MLP
from models.base_models import MLP3
import numpy as np
import random
import json
import argparse

global TORCH_BACKEND_DEVICE  # torch 后端设备，可能是cpu、cuda、mps
if torch.cuda.is_available():
    TORCH_BACKEND_DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    TORCH_BACKEND_DEVICE = 'mps'
else:
    TORCH_BACKEND_DEVICE = 'cpu'

parser = argparse.ArgumentParser(description="Model Pretrain", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--config', type=str, default='test_pretrain_template.json', help='config file e.g. test_pretrain.json')
parser.add_argument('--aligned_proportion', type=int, default=1000, help='aligned data proportions')
parser.add_argument('--use_global_zero', action='store_true')
parser.add_argument('--sup', action='store_true')
args = parser.parse_args()
try:
    with open(args.config, 'r') as f:
        config = json.load(f)
        np.random.seed(config['numpy_random_seed'])
        torch.manual_seed(config['torch_random_seed'])
        random.seed(config['random_seed'])

        comfort_aligned_except_classes = config['aligned_expect_classes']
        meta_params = config['meta_params']
        assert len(meta_params) == 4, "The number of meta_params must be 4"
except:
    print("Please provide a valid config file")
    print(config)
    exit(1)

def main(model, train_loader, val_loader, test_loader):
    for epoch in range(1, 101):
        model.train_model(train_loader=train_loader, epoch=epoch)
        model.evaluate_model(test_loader=val_loader, mode='val')
        if epoch % 10 == 0:
            model.evaluate_model(test_loader=test_loader, mode='test')


def split_features_for_clients(base_comfort_dataset, client_num):
    X = base_comfort_dataset.X
    total_columns = X.shape[1]
    columns_per_client = total_columns // client_num
    remainder = total_columns % client_num
    # 生成所有列索引并随机打乱
    all_columns = list(range(total_columns))
    # np.random.shuffle(all_columns)
    client_columns = []
    start_idx = 0
    for i in range(client_num):
        end_idx = start_idx + columns_per_client
        if i < remainder:
            end_idx += 1
        client_columns.append(all_columns[start_idx:end_idx])
        start_idx = end_idx
    client_columns = [[0, 1, 2, 3, 4, 5, 20, 21, 22, 23, 24, 25, 26, 27],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24],
                      [2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]
    return client_columns


def average_CRNet_models(client_list, weights=None):
    """
    对多个客户端的CRNET模型进行加权平均，使用第一个客户端的模型结构作为基准

    参数:
    - client_list: List[VFLClient], 客户端列表
    - weights: List[float], 每个客户端的权重，默认为等权重

    返回:
    - averaged_model: CRnetCls, 平均后的模型
    """
    # 检查客户端列表是否为空
    if not client_list:
        raise ValueError("Client list cannot be empty")

    # 获取第一个客户端的CRNET模型作为基准
    first_client = client_list[0]
    device = first_client.device

    # 深拷贝第一个客户端的模型结构
    averaged_model = copy.deepcopy(first_client.get_CRNet_model())
    averaged_model = averaged_model.to(device)

    # 如果没有提供权重，则使用等权重
    if weights is None:
        weights = [1.0 / len(client_list)] * len(client_list)
    else:
        # 确保权重数量与客户端数量相同
        if len(weights) != len(client_list):
            raise ValueError("Number of weights must match number of clients")
        # 归一化权重
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

    # 获取模型的状态字典
    state_dict = averaged_model.state_dict()

    # 对每个参数进行加权平均
    with torch.no_grad():
        for key in state_dict.keys():
            # 初始化参数累加器
            param_sum = torch.zeros_like(state_dict[key])

            # 累加每个客户端的加权参数
            for client, weight in zip(client_list, weights):
                client_param = client.get_CRNet_model().state_dict()[key]
                param_sum += weight * client_param

            # 更新平均后的参数
            state_dict[key] = param_sum

    # 将平均后的参数加载到模型中
    averaged_model.load_state_dict(state_dict)

    return averaged_model

if __name__ == '__main__':
    setup_logging(args=args)
    # args = prepare_exp()
    args_log = copy.deepcopy(args)
    args_log.col_names = ''
    logging.info(f"args = {vars(args_log)}")
    set_random_seed(42)
    comfort_class_dict = {i: str(i) for i in range(6)}
    # 定义每个客户端拥有的特征数量
    client_num = 3
    active_except_class = 0 if args.use_global_zero else 7
    num_classes = 6
    # unaligned_except_classes = [[1,2], [2,3], [1,3], [2,4]]
    # 参与方所拥有的类别：[1,2,3](主动方侧); [3,4,5]; [1,4,5];[1,2,5]; 对齐数据：[1,2] (每个类别数量：200)；第0类是全局不可见
    unaligned_except_classes = [[1, 2, 3, 4], [1, 2, 4, 5], [2, 4, 3, 5]] # 设置上第三个是主动方
    aligned_except_classes = [1, 4]
    # aligned_except_classes = comfort_aligned_except_classes
    base_comfort_dataset = ComfortDataset()
    feature_div = split_features_for_clients(base_comfort_dataset, client_num)
    base_test_datasets, aligned_client_datasets, unaligned_client_datasets, generated_client_datasets = (
        get_datasets_tab(base_dataset=base_comfort_dataset, client_num=client_num, feature_div=feature_div,
                     unaligned_expect_classes=unaligned_except_classes, aligned_expect_classes=aligned_except_classes,
                     global_unseen_class=active_except_class, num_aligned=args.aligned_proportion))
    if args.sup:
        label_type = 'sup'
    else:
        label_type = 'unsup'
    # 检查基础测试数据集
    assert base_test_datasets[0]['X'].shape[0] > 0, "Base test dataset should have samples"
    assert base_test_datasets[0]['Y'].shape[0] > 0, "Base test dataset should have labels"
    assert len(torch.unique(base_test_datasets[0]['Y'])) == num_classes, "Base test dataset should contain all classes"

    # 检查对齐的客户端数据集
    for dataset in aligned_client_datasets:
        assert dataset['X'].shape[0] > 0, "Aligned client dataset should have samples"
        print(dataset['X'].shape[1])
        assert dataset['Y'].shape[0] > 0, "Aligned client dataset should have labels"

    # 检查未对齐的客户端数据集
    for dataset in unaligned_client_datasets:
        assert dataset['X'].shape[0] > 0, "Unaligned client dataset should have samples"
        assert dataset['Y'].shape[0] > 0, "Unaligned client dataset should have labels"

    aligned_features = []
    test_features = []
    train_features = []
    aligned_labels = aligned_client_datasets[0]['Y']
    test_labels = base_test_datasets[0]['Y']
    client_gen_datas = []
    client_gen_labels = generated_client_datasets[0]['Y']
    client_aligned_datas = []
    client_unaligned_datas = []
    client_CRnets = []
    client_cluster_centers = []
    clients = []
    for i in range(client_num):
        print(f"start client {i} meta learning")
        table_dim = unaligned_client_datasets[i]['X'].size()[-1]
        print("table dimension", table_dim)
        client = VFLClient(aligned_dataset=aligned_client_datasets[i], unaligned_dataset=unaligned_client_datasets[i],
                           generated_dataset=generated_client_datasets[i], test_dataset=base_test_datasets[i],epochs=5, class_dict=comfort_class_dict,
                           table_dim=table_dim, meta_params=meta_params[i], client_id=i, label_mode=label_type)
        clients.append(client)
        client.train_base_model_aligned()
        # cluster_centers = client.train_self_CRNet()
        # # 元学习和元学习对应的特征提取
        client.meta_learning()
        # cluster_centers = client.train_self_CRNet()
        # aligned_features.append(client.extract_feature())
        test_features.append(client.extract_feature_test(base_test_datasets[i]))
        # train_features.append(client.extract_feature_test(aligned_client_datasets[i]))
        client_gen_datas.append(client.extract_feature_test(generated_client_datasets[i]))
        client_aligned_datas.append(client.extract_feature_test(aligned_client_datasets[i]))
        # client_unaligned_datas.append(client.extract_feature_unaligned())
        # client_CRnets.append(client.get_CRNet_model())
        # client_cluster_centers.append(cluster_centers)
        print(f"finish client {i} training")
    ensemble = EnsemblePredictor(clients, device='cuda')

    logging.info("Evaluating individual clients...")
    client_accuracies = []

    for i, client in enumerate(clients):
        # 准备验证数据集（使用aligned数据）
        # val_X = [
        #     torch.cat((client.unaligned_dataset['X'][j], client.aligned_dataset['X'][j]), dim=0)
        #     for j in range(len(client.unaligned_dataset['X']))
        # ]
        # if client.label_mode == 'sup':
        #     val_Y = torch.cat((client.unaligned_dataset['Y'], client.aligned_dataset['Y']), dim=0)
        # else:
        #     val_Y = client.pseudo_label

        # 创建验证数据集
        val_dataset = client.aligned_dataset
        # {
        #     'X': val_X,
        #     'Y': val_Y
        # }

        # 使用对应的测试数据集
        test_dataset = client.test_dataset

        # 评估单个客户端在验证集上的性能
        val_acc, val_class_accuracies = ensemble._evaluate_client(client, val_dataset)
        client_accuracies.append(val_acc)

        logging.info(f'Client [{i}] Validation Acc: {val_acc:.2f}% '
                     f'Per-class Val Accuracies: {val_class_accuracies}')

        # 评估单个客户端在测试集上的性能
        test_acc, test_class_accuracies = ensemble._evaluate_client(client, test_dataset)
        logging.info(f'Client [{i}] Test Acc: {test_acc:.2f}% '
                     f'Per-class Test Accuracies: {test_class_accuracies}')

    # 根据验证集性能设置权重
    accuracies = torch.tensor(client_accuracies)
    weights = F.softmax(accuracies / 10.0, dim=0).tolist()  # temperature=10.0
    ensemble.weights = weights

    logging.info("Client weights based on validation performance:")
    for i, weight in enumerate(weights):
        logging.info(f"Client {i}: {weight:.4f}")

    # 在每个客户端的测试集上评估集成模型
    logging.info("\nEvaluating ensemble model...")
    all_predictions = []
    all_labels = []

    for i, client in enumerate(clients):
        test_dataset = client.test_dataset
        test_loader = DataLoader(
            MultiViewDataset(test_dataset['X'], test_dataset['Y']),
            batch_size=32,
            shuffle=False
        )

        batch_predictions = []
        batch_labels = []

        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                batch_X = [X.to(ensemble.device) for X in batch_X]
                batch_Y = batch_Y.to(ensemble.device)

                # 获取所有客户端对这批数据的预测
                client_predictions = []
                for c in ensemble.clients:
                    c.base_model.eval()
                    outputs = c.base_model(batch_X, mode='client')
                    probs = F.softmax(outputs, dim=1)
                    client_predictions.append(probs)

                # 加权组合预测结果
                ensemble_pred = torch.zeros_like(client_predictions[0])
                for pred, weight in zip(client_predictions, ensemble.weights):
                    ensemble_pred += weight * pred

                batch_predictions.append(ensemble_pred)
                batch_labels.append(batch_Y)

        # 合并该客户端的所有批次预测结果
        predictions = torch.cat(batch_predictions)
        labels = torch.cat(batch_labels)

        # 计算该客户端测试集上的准确率
        _, predicted = torch.max(predictions, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = 100 * correct / total

        # 计算每个类的准确率
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        for label, prediction in zip(labels, predicted):
            class_total[label.item()] += 1
            if label == prediction:
                class_correct[label.item()] += 1

        class_accuracies = {cls: 100 * class_correct[cls] / class_total[cls]
        if class_total[cls] > 0 else 0.0
                            for cls in class_total.keys()}

        logging.info(f'Ensemble Model on Client {i} Test Data - '
                     f'Accuracy: {accuracy:.2f}% '
                     f'Per-class Accuracies: {class_accuracies}')

    # 保存结果
    results = {
        'client_weights': weights,
        'client_val_accuracies': client_accuracies,
        'ensemble_test_accuracies': accuracy,
        'ensemble_class_accuracies': class_accuracies
    }

    # 创建结果保存目录
    results_dir = os.path.join('results', 'ensemble')
    os.makedirs(results_dir, exist_ok=True)

    # 保存结果
    result_path = os.path.join(results_dir, 'ensemble_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)

    logging.info(f"Results saved to {result_path}")

