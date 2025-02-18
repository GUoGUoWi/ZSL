import gc
import os
import tempfile
import warnings

import math
import numpy as np
import pandas as pd
import torch
from PIL import Image
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, random_split, DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST


class UnsupportedError(Exception):
    def __init__(self, attr):
        super().__init__("Unsupported {}".format(attr))
        self.attr = attr


to_pil = transforms.ToPILImage()

class UnsupportedFormatError(UnsupportedError):
    def __init__(self):
        super().__init__("format")

class AdultRealDataset(Dataset):
    def __init__(self, datas, labels, transform=None):
        self.datas = datas
        # self.data2 = data2
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.datas[0])

    def __getitem__(self, index):
        label = self.labels[index]
        if self.transform is not None:
            imgs = [self.transform(data[index]) for data in self.datas]
        else:
            imgs = [data[index] for data in self.datas]
        return imgs, label


def get_credit_data(filepath=None):
    if filepath is None:
        df = pd.read_csv('./dataset/language/comfort/ashrae_db2.01.csv', low_memory=False)
    else:
        df = pd.read_csv(filepath, low_memory=False)
    df = df.loc[:, df.isna().mean() < .67]
    df = df[['Season', 'Koppen climate classification', 'Building type',
             'Cooling startegy_building level', 'Clo', 'Met',
             'Air temperature (C)', 'Relative humidity (%)',
             'Air velocity (m/s)', 'Outdoor monthly air temperature (C)', 'Thermal comfort']]
    df = df.dropna()
    # Checking for outliers in the target variable "Thermal Comfort"
    df = df[df['Thermal comfort'] != 'Na']
    df = df[df['Thermal comfort'] != '1.3']
    df = pd.concat(
        [df.select_dtypes([], ['object']), df.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')],
        axis=1)
    # Converting the data to integer so 5.0 and 5 will under one category and outliers would be categorised in nearest integer
    df['Thermal comfort'] = df['Thermal comfort'].astype('float')
    df['Thermal comfort'] = df['Thermal comfort'].astype('int')
    df = df.drop_duplicates()
    # Converting the thermal comfort column back to categorical data type to make it cleaner and easier for prediction
    # df['Thermal comfort'] = df['Thermal comfort'].astype('category')
    y = df['Thermal comfort']
    df.drop(['Thermal comfort'], axis=1, inplace=True)
    # One hot encoding
    # 分离数值列和字符串列
    numerical_data = df.iloc[:, :6].values
    categorical_data = df.iloc[:, 6:]

    # 使用OneHotEncoder对字符串列进行编码
    encoder = OneHotEncoder(sparse_output=False)
    encoded_categorical_data = encoder.fit_transform(categorical_data)
    # 打印每个列在OneHotEncoder编码后的列索引范围
    start_idx = 0
    for col, categories in zip(categorical_data.columns, encoder.categories_):
        end_idx = start_idx + len(categories)
        print(f"Column '{col}' encoded into columns {start_idx} to {end_idx - 1}")
        start_idx = end_idx
    # 拼接数值和编码后的字符串特征
    combined_data = np.hstack((numerical_data, encoded_categorical_data))
    X = combined_data
    # df = pd.get_dummies(data=df, drop_first=True)
    #X = df.values
    y.values[:] = y.values - 1
    y = y.values
    return X, y


def split_train_data(features, labels, aligned_num=200, train_proportion=0.8):
    # 合并特征和标签成一个数组
    data = np.column_stack((features, labels.reshape(-1, 1)))
    # 打乱数组
    np.random.shuffle(data)
    # 切分数组为三个部分
    split_size = int((data.shape[0] - aligned_num) * train_proportion)+aligned_num
    split_data = np.split(data, [aligned_num, split_size])
    # 分别提取特征和标签
    split_features = [part[:, :-1] for part in split_data]
    split_labels = [part[:, -1] for part in split_data]
    return split_features, split_labels


def split_feature(features):
    # 切分特征数组
    split_features = np.array_split(features, 4, axis=1)
    # 创建张量列表，并用 0 进行填充
    tensor_list = []
    for feature_chunk in split_features:
        feature_chunk = feature_chunk.astype(np.float32)
        padded_chunk = np.pad(feature_chunk, ((0, 0), (0, 14 - feature_chunk.shape[1])), mode='constant',
                              constant_values=0)
        tensor_list.append(torch.tensor(padded_chunk))
    return tensor_list


def vertical_split_image(x, num_parties, target_size=(32, 32)):
    # 这里输入的要是张量
    if len(x.shape) != 4:
        print("Wrong format of image, got {}".format(str(x.shape)))
        raise UnsupportedFormatError

    # 首先将格式从 (N, H, W, C) 转换为 (N, C, H, W)
    x = x.permute(0, 3, 1, 2)

    m, n = x.shape[2], x.shape[3]  # 注意：现在高度和宽度在第2、3维度
    a, b = get_closest_factor(num_parties)

    if a != b:
        warnings.warn("num_parties is recommended to be perfect square numbers. a={}, b={}".format(a, b))
    if m % a != 0 or n % b != 0:
        warnings.warn("The image size for each party may be not equal. m={}, n={}, a={}, b={}".format(m, n, a, b))

    xs = []
    for i in range(a):
        for j in range(b):
            if i != m - 1 and j != n - 1:
                x_i_j = x[:, :, i * m // a: (i + 1) * m // a, j * n // b: (j + 1) * n // b]
            elif i == m - 1 and j != n - 1:
                x_i_j = x[:, :, i * m // a:, j * n // b: (j + 1) * n // b]
            elif i != m - 1 and j == n - 1:
                x_i_j = x[:, :, i * m // a: (i + 1) * m // a, j * n // b:]
            else:
                x_i_j = x[:, :, i * m // a:, j * n // b:]

            # 使用插值调整大小
            x_i_j_resized = F.interpolate(x_i_j, size=target_size, mode='bilinear', align_corners=False)

            # 将格式转回 (N, H, W, C)
            x_i_j_resized = x_i_j_resized.permute(0, 2, 3, 1)

            xs.append(x_i_j_resized)

    return xs

def get_closest_factor(n: int):
    """
    find the two closest integers a & b which, when multiplied, equal to n
    :param n: integer
    :return: a, b
    """
    a = math.ceil(math.sqrt(n))
    while True:
        if n % a == 0:
            b = n // a
            return a, b
        a -= 1


def get_comfort_dataset(client_num, aligned_proportion, sel_idx, fp=None):
    # sel_idx = [[0,2], [1,3], [0,3], [1,2]] # [[0], [1], [2], [3]]# [[0,2], [1,3], [0,3], [1,2]]
    # sel_idx = [[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]]
    num_parties = client_num
    adult_dataX, adult_dataY = get_credit_data(fp)
    split_features, split_labels = split_train_data(adult_dataX, adult_dataY, aligned_num=int(aligned_proportion))
    aligned_samples, unaligned_train_samples, test_samples = split_features
    aligned_train_labels, unaligned_train_label, test_labels = split_labels
    aligned_train_labels, unaligned_train_label, test_labels = (torch.tensor(aligned_train_labels.astype(np.int64)),
                                                                torch.tensor(unaligned_train_label.astype(np.int64)),
                                                                torch.tensor(test_labels.astype(np.int64)))
    aligned_train_datasets = split_feature(aligned_samples)
    aligned_train_datasets = [[aligned_train_datasets[i] for i in idxs] for idxs in sel_idx]
    # aligned_train_datasets = [[aligned_train_datasets[i], aligned_train_datasets[j]] for i, j in sel_idx]

    unaligned_train_datas = split_feature(unaligned_train_samples)
    # unaligned_train_datasets = [FmnistRealDataset(data1=unaligned_train_datas[i], data2=unaligned_train_datas[j], labels=unaligned_train_label,
    #                             transform=transform) for i, j in sel_idx]
    unaligned_train_datasets = [AdultRealDataset(datas=[unaligned_train_datas[i] for i in idxs], labels=unaligned_train_label,
                                ) for idxs in sel_idx]

    test_datasets = split_feature(test_samples)
    # test_datasets = [[test_datasets[i], test_datasets[j]] for i, j in sel_idx]
    test_datasets = [[test_datasets[i] for i in idxs] for idxs in sel_idx]
    # test_datasets = [FmnistRealDataset(data=test_data, labels=test_labels, transform=transform) for test_data
    #                  in test_datas]
    return aligned_train_datasets, aligned_train_labels, unaligned_train_datasets, test_datasets, test_labels


class MultiViewFMNIST(Dataset):
    def __init__(self, data_type='train', transform=None, max_samples=10000):
        """
        多视角Fashion-MNIST数据集，限制处理的数据量

        Parameters:
        data_type: str, 'train' 或 'test'
        transform: Optional[callable], 数据预处理方法
        max_samples: int, 最大处理的样本数量
        """
        self.transform = transform
        self.X = []  # 将存储4个视角的数据
        self.y = None
        self.max_samples = max_samples

        # 加载原始FMNIST数据
        self.fmnist = FashionMNIST(
            root='../VFL-Protos/dataset',
            train=(data_type == 'train'),
            transform=None,
            download=True
        )

        # 获取类别信息
        self.classes = self.fmnist.classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.class_num = len(self.classes)

        # 数据预处理
        self._process_data()

    def _process_data(self):
        """处理数据集，创建4个视角版本，限制数据量"""
        # 限制数据量
        fmnist_data = self.fmnist.data[:self.max_samples].unsqueeze(-1).float() / 255.0
        fmnist_targets = self.fmnist.targets[:self.max_samples]

        # 创建4个视角
        split_views = vertical_split_image(fmnist_data, 4)

        # 初始化4个视角的列表
        self.X = [[] for _ in range(4)]

        # 分批处理数据以减少内存使用
        batch_size = 1000
        for start_idx in range(0, len(fmnist_data), batch_size):
            end_idx = min(start_idx + batch_size, len(fmnist_data))

            # 处理每个视角
            for view_idx, view_data in enumerate(split_views):
                view_batch = view_data[start_idx:end_idx]
                batch_images = []

                for img in view_batch:
                    pil_img = Image.fromarray(img.squeeze().numpy() * 255).convert('L')
                    if self.transform:
                        pil_img = self.transform(pil_img)
                    batch_images.append(pil_img)

                # 将批次数据添加到对应视角
                self.X[view_idx].extend(batch_images)

            # 清理中间变量
            del view_batch, batch_images
            gc.collect()

        # 将列表转换为张量
        for view_idx in range(4):
            self.X[view_idx] = torch.stack(self.X[view_idx])

        # 转换标签为张量
        self.y = fmnist_targets

        # 验证数据
        self._verify_data()

    def _verify_data(self):
        """验证数据的一致性"""
        n_samples = len(self.y)
        for view_idx, view_data in enumerate(self.X):
            assert len(view_data) == n_samples, \
                f"视角 {view_idx} 的样本数量不匹配: {len(view_data)} != {n_samples}"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        """获取指定索引的多视角数据"""
        return [view_data[index] for view_data in self.X], self.y[index]


class MultiViewFMNIST_(Dataset):
    def __init__(self, data_type='train', transform=None):
        """
        多视角Fashion-MNIST数据集

        Parameters:
        data_type: str, 'train' 或 'test'
        transform: Optional[callable], 数据预处理方法
        """
        self.transform = transform
        self.X = []  # 将存储4个视角的数据
        self.y = None

        # 加载原始FMNIST数据
        self.fmnist = FashionMNIST(
            root='../VFL-Protos/dataset',
            train=(data_type == 'train'),
            transform=None,
            download=True
        )

        # 获取类别信息
        self.classes = self.fmnist.classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.class_num = len(self.classes)

        # 数据预处理
        self._process_data()

    def _process_data(self):
        """处理数据集，创建4个视角版本"""
        # 转换数据格式 [N, H, W] -> [N, H, W, 1]
        fmnist_data = self.fmnist.data.unsqueeze(-1).float() / 255.0
        # print(fmnist_data.shape)
        # 创建4个视角
        split_views = vertical_split_image(fmnist_data, 4)  # 分成4个部分

        # 初始化4个视角的列表
        for _ in range(4):
            self.X.append([])

        # 将数据组织成与ModelNet相同的格式
        for view_idx, view_data in enumerate(split_views):
            # 转换为PIL Image并应用transform
            view_images = []
            for img in view_data:
                # print(img.shape)
                # [H, W, 1] -> PIL Image
                pil_img = Image.fromarray(img.squeeze().numpy() * 255).convert('L')
                if self.transform:
                    pil_img = self.transform(pil_img)
                view_images.append(pil_img)

            # 堆叠成张量
            self.X[view_idx] = torch.stack(view_images)

        # 转换标签为张量
        self.y = torch.tensor(self.fmnist.targets)

        # 验证数据
        self._verify_data()

    def _verify_data(self):
        """验证数据的一致性"""
        n_samples = len(self.y)
        for view_idx, view_data in enumerate(self.X):
            assert len(view_data) == n_samples, \
                f"视角 {view_idx} 的样本数量不匹配: {len(view_data)} != {n_samples}"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        """获取指定索引的多视角数据"""
        return [view_data[index] for view_data in self.X], self.y[index]


class FmnistDataset(Dataset):
    def __init__(self, X=None, y=None):
        if X is None or y is None:
            X, y = get_credit_data()
        self.X = torch.tensor(X.astype(np.float32), dtype=torch.float32)
        self.y = torch.tensor(y.astype(np.int64), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_class_indices(self, class_label):
        indices = (self.y == class_label).nonzero(as_tuple=True)[0]
        return indices.tolist()

    def split_and_select_features(self, X, split_num, feature_list):
        # X的形状为 [num_samples, num_features]
        num_samples, num_features = X.shape

        # 计算需要填充的长度
        pad_size = (split_num - (num_features % split_num)) % split_num
        if pad_size > 0:
            padding = torch.zeros((num_samples, pad_size), dtype=X.dtype)
            X = torch.cat((X, padding), dim=1)

        # 更新填充后的特征数目
        num_features = X.shape[1]
        split_size = num_features // split_num
        splits = torch.split(X, split_size, dim=1)
        selected_splits = [splits[i] for i in feature_list]
        X2 = torch.cat(selected_splits, dim=1)
        return X2

    def create_features_with_labels(self, class_labels):
        mask = torch.isin(self.y, torch.tensor(class_labels))
        X1 = self.X[mask]
        y1 = self.y[mask]
        return X1, y1

    def create_features_with_features(self, X1, y1, split_num, feature_list):
        X2 = self.split_and_select_features(X1, split_num, feature_list)
        return X2, y1

    def create_subset_with_labels(self, class_labels):
        mask = torch.isin(self.y, torch.tensor(class_labels))
        X1 = self.X[mask]
        y1 = self.y[mask]
        return FmnistDataset(X1.numpy(), y1.numpy())

    def create_subset_with_features(self, X1, y1, split_num, feature_list):
        X2 = self.split_and_select_features(X1, split_num, feature_list)
        return FmnistDataset(X2.numpy(), y1.numpy())

    def train_test_split(self, test_size=0.2):
        total_size = len(self)
        if test_size < 1:
            test_size = int(total_size * test_size)
        else:
            test_size = int(test_size)
        train_size = total_size - test_size
        np.random.seed(42)
        indices = np.random.permutation(total_size)
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        dataset_a = FmnistDataset(X=self.X.numpy()[train_idx],y=self.y.numpy()[train_idx])
        dataset_b = FmnistDataset(X=self.X.numpy()[test_idx],y=self.y.numpy()[test_idx])
        return dataset_a, dataset_b