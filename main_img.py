import copy
import logging

import torch
from torchvision.transforms import transforms

from client.client_template_img import VFLClient
from client.server_template import VFLServer
from data.comfort_dataset import ComfortDataset
from data.modelnet_dataset import MultiViewModelNet
from etc.prepare_exp import get_datasets_tab, get_datasets_img, get_datasets_img_transform
import numpy as np
import random
import json
import argparse

from etc.utils import setup_logging, set_random_seed

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
parser.add_argument('--sup', action='store_true') # type=bool, default=True)
parser.add_argument('--meta', action='store_true')
parser.add_argument('--fine',action='store_true')

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


if __name__ == '__main__':
    setup_logging(args=args)
    args_log = copy.deepcopy(args)
    args_log.col_names = ''
    logging.info(f"args = {vars(args_log)}")
    comfort_class_dict = {i: str(i) for i in range(6)}
    # 基本设置
    client_num = 4
    num_classes = 10
    active_except_class = 7  if args.use_global_zero else 11# 全局不可见类别
    set_random_seed(42)
    # 设置两个对齐类别（所有客户端都拥有的类别）
    aligned_except_classes = [2, 5]

    # 为每个客户端设置未对齐类别（不包括对齐类别和全局不可见类别）
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
    data_root = '/root/FEZE/dataset/modelnet_aligned_new'# 'dataset/image/modelnet_aligned_new'
    # aligned_except_classes = comfort_aligned_except_classes
    base_comfort_dataset = MultiViewModelNet(data_dir=data_root, data_type='train', transform=gray_test_transform)
    classes, _ = base_comfort_dataset.find_class(data_root)
    print(classes)
    print(1)
    feature_div = [[0], [3], [6], [9]]
        # [[0,1,2,3], [3,4,5,6], [6,7,8,10], [10, 11, 4, 0]]#split_features_for_clients(base_comfort_dataset, client_num)
    if args.sup:
        label_type = 'sup'
    else:
        label_type = 'unsup'
    base_test_datasets, aligned_client_datasets, unaligned_client_datasets, generated_client_datasets = (
        get_datasets_img_transform(base_dataset=base_comfort_dataset, client_num=client_num, feature_div=feature_div,
                         unaligned_expect_classes=unaligned_except_classes, aligned_expect_classes=aligned_except_classes,
                         global_unseen_class=active_except_class, num_aligned=args.aligned_proportion,
                                   train_transform=None, test_transform=None)) # augmentation_transform
    # # 检查基础测试数据集
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
    client_gen_labels = base_test_datasets[0]['Y']
    client_aligned_datas = []
    client_unaligned_datas = []
    client_CRnets = []
    client_cluster_centers = []
    client_class_prototypes = []
    clients = []
    for i in range(client_num):
        print(f"start client {i} meta learning")
        # table_dim = unaligned_client_datasets[i]['X'].size()[-1]
        # print("table dimension", table_dim)
        client = VFLClient(aligned_dataset=aligned_client_datasets[i], unaligned_dataset=unaligned_client_datasets[i],
                           generated_dataset=base_test_datasets[i], epochs=5, class_dict=comfort_class_dict,
                           table_dim=7, meta_params=meta_params[i], client_id=i, label_mode=label_type,
                           meta=args.meta, fine=args.fine)
        clients.append(client)
        logging.info('start meta learning')
        # 元学习和元学习对应的特征提取
        client.meta_learning()
        logging.info('start training CRNET')
        cluster_centers, class_prototypes = client.train_self_CRNet_batch()
        aligned_features.append(client.extract_feature())
        test_features.append(client.extract_feature_test(base_test_datasets[i]))
        train_features.append(client.extract_feature_test(aligned_client_datasets[i]))
        client_gen_datas.append(client.extract_feature_gen())
        client_aligned_datas.append(client.extract_feature_aligned())
        client_unaligned_datas.append(client.extract_feature_unaligned())
        client_CRnets.append(client.get_CRNet_model())
        client_cluster_centers.append(cluster_centers)
        client_class_prototypes.append(class_prototypes)
        print(f"finish client {i} training")

    dim = (aligned_features[0].shape[1] * client_num)
    server = VFLServer(input_dim=dim, aligned_test_features=client_gen_datas, aligned_test_labels=client_gen_labels,
                       aligned_train_features=train_features, aligned_train_labels=aligned_labels, num_client=4,
                       unaligned_features=client_unaligned_datas[0], unaligned_labels=unaligned_client_datasets[0]['Y'],
                       num_labels=num_classes, except_classes=aligned_except_classes+[0], dataset='modelnet') # except class是对其数据和全局不可见数据，第0类是全局不可见
    global_client_models, _ = server.CRNet_progress_img(client_CRnets, client_cluster_centers, client_class_prototypes, client_gen_datas, client_gen_labels)
    CR_epochs = 2
    for epoch in range(CR_epochs):
        client_CRnets = []
        client_cluster_centers = []
        client_class_prototypes = []
        for i, client in enumerate(clients):
            print(f'--- Round {i}, Client {i} ---')
            client.CRNet = global_client_models[i]
            cluster_centers, class_prototypes = client.train_self_CRNet_batch()
            client_CRnets.append(client.get_CRNet_model())
            client_cluster_centers.append(cluster_centers)
            client_class_prototypes.append(class_prototypes)
        global_client_models, _ = server.CRNet_progress_img(client_CRnets, client_cluster_centers, client_class_prototypes, client_gen_datas, client_gen_labels)



