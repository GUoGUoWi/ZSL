import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, random_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


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


class ComfortDataset(Dataset):
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
        return ComfortDataset(X1.numpy(), y1.numpy())

    def create_subset_with_features(self, X1, y1, split_num, feature_list):
        X2 = self.split_and_select_features(X1, split_num, feature_list)
        return ComfortDataset(X2.numpy(), y1.numpy())

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
        dataset_a = ComfortDataset(X=self.X.numpy()[train_idx],y=self.y.numpy()[train_idx])
        dataset_b = ComfortDataset(X=self.X.numpy()[test_idx],y=self.y.numpy()[test_idx])
        return dataset_a, dataset_b