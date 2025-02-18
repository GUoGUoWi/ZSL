import numpy as np
import torch
from torch.utils.data import Dataset, random_split


class BaseDataset(Dataset):
    def __init__(self, X, y):
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

    def create_subset_with_splits(self, class_labels, split_num, feature_list):
        mask = torch.isin(self.y, torch.tensor(class_labels))
        X1 = self.X[mask]
        y1 = self.y[mask]

        X2 = self.split_and_select_features(X1, split_num, feature_list)
        return BaseDataset(X2.numpy(), y1.numpy())

    def train_test_split(self, test_size=0.2):
        total_size = len(self)
        test_size = int(total_size * test_size)
        train_size = total_size - test_size
        return random_split(self, [train_size, test_size])
