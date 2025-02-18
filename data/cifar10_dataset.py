import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from data.fmnist_dataset import vertical_split_image, AdultRealDataset


class MultiViewCIFAR10(Dataset):
    def __init__(self, data_type='train', transform=None):
        """
        多视角CIFAR10数据集

        Parameters:
        data_type: str, 'train' 或 'test'
        transform: Optional[callable], 数据预处理方法
        """
        self.transform = transform
        self.X = []  # 存储4个视角的数据
        self.y = None

        # 加载原始CIFAR10数据
        self.cifar = CIFAR10(
            root='./dataset',
            train=(data_type == 'train'),
            transform=None,
            download=True
        )

        # 获取类别信息
        self.classes = self.cifar.classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.class_num = len(self.classes)

        # 数据预处理
        self._process_data()

    def _process_data(self):
        """处理数据集，创建4个视角版本"""
        # 转换数据格式 [N, H, W, C]
        cifar_data = torch.tensor(np.array(self.cifar.data)).float() / 255.0

        # 创建4个视角
        split_views = vertical_split_image(cifar_data, 4)  # 分成4个部分

        # 初始化4个视角的列表
        for _ in range(4):
            self.X.append([])

        # 将数据组织成所需格式
        for view_idx, view_data in enumerate(split_views):
            view_images = []
            for img in view_data:
                # [H, W, 3] -> PIL Image
                pil_img = Image.fromarray((img.numpy() * 255).astype(np.uint8))
                if self.transform:
                    pil_img = self.transform(pil_img)
                view_images.append(pil_img)

            # 堆叠成张量
            self.X[view_idx] = torch.stack(view_images)

        # 转换标签为张量
        self.y = torch.tensor(self.cifar.targets)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        """获取指定索引的多视角数据"""
        return [view_data[index] for view_data in self.X], self.y[index]


def get_cifar10_dataset(client_num, aligned_proportion, sel_idx, transform=None):
    """
    获取CIFAR10数据集的联邦学习版本

    Args:
        client_num: 客户端数量
        aligned_proportion: 对齐数据的比例
        sel_idx: 特征选择索引
        transform: 数据转换
    """
    # 默认转换
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # 加载训练集和测试集
    train_dataset = MultiViewCIFAR10('train', transform=transform)
    test_dataset = MultiViewCIFAR10('test', transform=transform)

    # 计算对齐样本数量
    aligned_num = int(len(train_dataset) * aligned_proportion)

    # 分割训练数据为对齐和非对齐部分
    total_indices = list(range(len(train_dataset)))
    np.random.shuffle(total_indices)

    aligned_indices = total_indices[:aligned_num]
    unaligned_indices = total_indices[aligned_num:]

    # 创建数据子集
    def create_subset(dataset, indices, sel_idx):
        X = [dataset.X[i][indices] for i in range(len(dataset.X))]
        y = dataset.y[indices]
        return [[X[i] for i in idxs] for idxs in sel_idx], y

    # 创建对齐数据集
    aligned_train_datasets, aligned_train_labels = create_subset(
        train_dataset, aligned_indices, sel_idx)

    # 创建非对齐数据集
    unaligned_train_datasets = []
    for idxs in sel_idx:
        X = [train_dataset.X[i][unaligned_indices] for i in idxs]
        y = train_dataset.y[unaligned_indices]
        unaligned_train_datasets.append(
            AdultRealDataset(datas=X, labels=y))

    # 创建测试数据集
    test_datasets, test_labels = create_subset(
        test_dataset, list(range(len(test_dataset))), sel_idx)

    return (aligned_train_datasets, aligned_train_labels,
            unaligned_train_datasets, test_datasets, test_labels)
