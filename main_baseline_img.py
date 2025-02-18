import copy
import logging

import torch
from torchvision import transforms

from client.client_template_img import VFLClient

from data.comfort_dataset import ComfortDataset
from data.modelnet_dataset import MultiViewModelNet
from etc.prepare_exp import get_datasets_tab, get_datasets_img_transform
from etc.utils import setup_logging, set_random_seed, calculate_tensors_size_detailed
from models.CRNet.models_cls import CRnetCls
from models.CRNet.train_CRNet_cls import train_and_eval_CRNet, train_and_eval_MLP
from models.base_models import MLP3
import numpy as np
import random
import json
import argparse

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
parser.add_argument('--dataset', type=str, default='modelnet', help='dataset to choose')
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
    # 基本设置
    client_num = 4
    num_classes = 10
    active_except_class = 7 if args.use_global_zero else 11  # 全局不可见类别

    # 设置两个对齐类别（所有客户端都拥有的类别）
    aligned_except_classes = [2, 5]

    # 为每个客户端设置未对齐类别（不包括对齐类别和全局不可见类别）a
    # 每个客户端分配4个未对齐类别
    unaligned_except_classes = [
        [0, 1, 3, 4, 8],  # 客户端0的未对齐类别
        [1, 3, 4, 8, 9],  # 客户端1的未对齐类别
        [0, 1, 4, 8, 9],  # 客户端2的未对齐类别
        [0, 3, 4, 6, 9]  # 客户端3(主动方)的未对齐类别
    ]
    gray_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    gray_mild_train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mild_train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 首先调整大小
        transforms.RandomCrop(224),  # 随机裁剪
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 颜色抖动
        transforms.Grayscale(num_output_channels=3),  # 转换为3通道
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.485, 0.485],  # ImageNet的灰度图归一化值
                             std=[0.229, 0.229, 0.229])
    ])
    augmentation_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    unaligned_except_classes = [classes + aligned_except_classes for classes in unaligned_except_classes]
    data_root = './dataset/modelnet_aligned_new' # 'dataset/image/modelnet_aligned_new'
    # aligned_except_classes = comfort_aligned_except_classes
    base_comfort_dataset = MultiViewModelNet(data_dir=data_root, data_type='train', transform=gray_test_transform)
    classes, _ = base_comfort_dataset.find_class(data_root)
    print(classes)
    print(1)
    if args.sup:
        label_type = 'sup'
    else:
        label_type = 'unsup'
    feature_div = [[0], [3], [6], [9]]#  [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 10],
                  # [10, 11, 4, 0]]  # split_features_for_clients(base_comfort_dataset, client_num)
    base_test_datasets, aligned_client_datasets, unaligned_client_datasets, generated_client_datasets = (
        get_datasets_img_transform(base_dataset=base_comfort_dataset, client_num=client_num, feature_div=feature_div,
                                   unaligned_expect_classes=unaligned_except_classes,
                                   aligned_expect_classes=aligned_except_classes,
                                   global_unseen_class=active_except_class, num_aligned=args.aligned_proportion,
                                   train_transform=None, test_transform=None))
    # 检查基础测试数据集
    # assert base_test_datasets[0]['X'].shape[0] > 0, "Base test dataset should have samples"
    # assert base_test_datasets[0]['Y'].shape[0] > 0, "Base test dataset should have labels"
    # assert len(torch.unique(base_test_datasets[0]['Y'])) == num_classes, "Base test dataset should contain all classes"
    #
    # # 检查对齐的客户端数据集
    # for dataset in aligned_client_datasets:
    #     assert dataset['X'].shape[0] > 0, "Aligned client dataset should have samples"
    #     print(dataset['X'].shape[1])
    #     assert dataset['Y'].shape[0] > 0, "Aligned client dataset should have labels"
    #
    # # 检查未对齐的客户端数据集
    # for dataset in unaligned_client_datasets:
    #     assert dataset['X'].shape[0] > 0, "Unaligned client dataset should have samples"
    #     assert dataset['Y'].shape[0] > 0, "Unaligned client dataset should have labels"

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
        # table_dim = unaligned_client_datasets[i]['X'].size()[-1]
        # print("table dimension", table_dim)
        client = VFLClient(aligned_dataset=aligned_client_datasets[i], unaligned_dataset=unaligned_client_datasets[i],
                           generated_dataset=generated_client_datasets[i], test_dataset=base_test_datasets[i],epochs=5, class_dict=MODELNET_DESCRIPTIONS_SEEN,
                           table_dim=8, meta_params=meta_params[i], client_id=i, dataset_type=args.dataset, label_mode=label_type, meta=False)
        clients.append(client)
        client.train_base_model_aligned()
        # cluster_centers = client.train_self_CRNet()
        # # 元学习和元学习对应的特征提取
        client.meta_learning()
        cluster_centers = client.train_self_CRNet()
        # aligned_features.append(client.extract_feature())
        test_features.append(client.extract_feature_test(base_test_datasets[i]))
        # train_features.append(client.extract_feature_test(aligned_client_datasets[i]))
        client_gen_datas.append(client.extract_feature_test(generated_client_datasets[i]))
        client_aligned_datas.append(client.extract_feature_test(aligned_client_datasets[i]))
        # client_unaligned_datas.append(client.extract_feature_unaligned())
        # client_CRnets.append(client.get_CRNet_model())
        # client_cluster_centers.append(cluster_centers)
        print(f"finish client {i} training")
    X_test = torch.cat(test_features, dim=1)
    y_test = test_labels
    X_aligned = torch.cat(client_aligned_datas, dim=1)
    X_generated = torch.cat(client_gen_datas, dim=1)
    #GEN部分长度不一
    X_train = torch.cat([X_generated, X_aligned], dim=0)
    y_train = torch.cat([client_gen_labels, aligned_labels], dim=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_CR = CRnetCls(num_experts=3, semantic_dim=200, visual_dim=128 * 4, hidden_dim=100).to(device)
    idx_to_label = {
            0: "very uncomfortable",
            1: "uncomfortable",
            2: "slightly uncomfortable",
            3: "slightly comfortable",
            4: "comfortable",
            5: "very comfortable"
        }
    commu_tensors = [X_train, y_train, X_test, y_test]
    commu_info = calculate_tensors_size_detailed(commu_tensors)
    logging.info(f'Communication Cost: {commu_info}')
    train_and_eval_CRNet(model_CR, X_train, y_train, X_test, y_test,
                         idx_to_label, num_epochs=20, learning_rate=1e-3, device='cuda')

    model_MLP = MLP3(input_dim=128*4).to(device)
    train_and_eval_MLP(model_MLP, X_train, y_train, X_test, y_test,
                         idx_to_label, num_epochs=20, learning_rate=1e-3, device='cuda')
    # model.to('cuda')
    # global_CRNet = average_CRNet_models(clients)
    # for test_feature in test_features:
    #     eval_CRNet(global_CRNet, test_feature, test_labels)


    # dim = (aligned_features[0].shape[1] * client_num)
    # server = VFLServer(input_dim=dim, aligned_test_features=client_gen_datas, aligned_test_labels=client_gen_labels,
    #                    aligned_train_features=train_features, aligned_train_labels=aligned_labels,
    #                    unaligned_features=client_unaligned_datas[0], unaligned_labels=unaligned_client_datasets[0]['Y'],
    #                    num_labels=num_classes, except_classes=aligned_except_classes+[0]) # except class是对其数据和全局不可见数据，第0类是全局不可见
    # server.CRNet_progress(client_CRnets, client_cluster_centers, client_gen_datas, client_gen_labels)
    # global_CRNet = server.get_CRNet_model()
    # CR_epochs = 1
    # for epoch in range(CR_epochs):
    #     client_CRnets = []
    #     client_cluster_centers = []
    #     for i, client in enumerate(clients):
    #         print(f'--- Round {i}, Client {i} ---')
    #         client.CRNet = global_CRNet
    #         cluster_centers = client.train_self_CRNet()
    #         client_CRnets.append(client.get_CRNet_model())
    #         aligned_features.append(client.extract_feature())
    #         client_gen_datas.append(client.extract_feature_gen())
    #         client_cluster_centers.append(cluster_centers)
    #     server.CRNet_progress(client_CRnets, client_cluster_centers, client_gen_datas, client_gen_labels)


