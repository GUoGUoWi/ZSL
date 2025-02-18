import os

import torch
from PIL import Image




class MultiViewModelNet:
    def __init__(self, data_dir, data_type, transform=None):
        """
        Parameters:
        data_dir: 数据根目录
        data_type: 数据类型（train/test）
        transform: 数据预处理方法
        """
        # DATA_DIR = '../VFL-Protos/dataset/modelnet_aligned_new/'
        # data_type = 'train' or 'test'
        self.transform = transform
        self.X = []  # 将存储12个视角的数据
        self.y = None

        # 首先获取类别信息
        self.classes, self.class_to_idx = self.find_class(data_dir)
        self.class_num = len(self.classes)

        # 加载数据
        self._load_data(data_dir, data_type)

    def find_class(self, dir):
        """查找所有类别"""
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _load_data(self, data_dir, data_type):
        """加载多视角数据"""
        # 生成12个视角的后缀
        subfixes = [str(((i - 1) * 30)).zfill(3) + '_' + str(i).zfill(3)
                    for i in range(1, 13)]

        # 收集所有样本的路径和标签
        all_samples = []
        all_labels = []

        for label in self.classes:
            all_files = [d for d in os.listdir(os.path.join(data_dir, label, data_type))]

            # 获取所有样本的索引
            if len(label.split('_')) == 1:
                all_indexes = list(set([item.split('_')[1] for item in all_files]))
            else:
                all_indexes = list(set([item.split('_')[2] for item in all_files]))

            for ind in all_indexes:
                cls = '{}_{}'.format(label, ind)
                # 获取该样本的12个视角
                all_views = ['{}/{}.{}.png'.format(cls, cls[:-4], str(int(sg_subfix[-1]) + 1))
                             for sg_subfix in subfixes]
                all_views = [os.path.join(data_dir, label, data_type, item)
                             for item in all_views]

                all_samples.append(all_views)
                all_labels.append(self.class_to_idx[label])

        # 将数据转换为所需格式
        # 初始化12个视角的列表
        for _ in range(12):
            self.X.append([])

        # 加载所有图像数据
        for sample_paths in all_samples:
            for view_idx, img_path in enumerate(sample_paths):
                img = Image.open(img_path).convert('L')
                if self.transform:
                    img = self.transform(img)
                self.X[view_idx].append(img)

        # 将每个视角的图像数据堆叠成张量
        for view_idx in range(12):
            self.X[view_idx] = torch.stack(self.X[view_idx])

        # 转换标签为张量
        self.y = torch.tensor(all_labels)

        # 验证数据
        self._verify_data()

    def _verify_data(self):
        """验证数据的一致性"""
        n_samples = len(self.y)
        for view_data in self.X:
            assert len(view_data) == n_samples, \
                f"视角数据样本数量不一致: {len(view_data)} != {n_samples}"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        """获取指定索引的多视角数据"""
        return [view_data[index] for view_data in self.X], self.y[index]


class MultiViewDataset():
    def __init__(self, views_data, labels=None, transform=None):
        """
        初始化多视角数据集
        :param views_data: 列表的列表，每个子列表包含一个视角的所有样本数据
                         形状为 [num_views][num_samples, channels, height, width]
        :param labels: 标签数据，形状为 [num_samples]
        :param transform: 数据转换函数（可选）
        """
        self.views = [torch.FloatTensor(view) if not torch.is_tensor(view) else view
                      for view in views_data]
        self.y = torch.LongTensor(labels) if labels is not None and not torch.is_tensor(labels) else labels
        self.transform = transform

        # 验证所有视角的样本数量一致
        num_samples = len(self.views[0])
        assert all(len(view) == num_samples for view in self.views), "所有视角的样本数量必须相同"

        if self.y is not None:
            assert len(self.y) == num_samples, "标签数量必须与样本数量相同"

    def __len__(self):
        return len(self.views[0])

    def __getitem__(self, idx):
        # 获取所有视角的第idx个样本
        sample_views = [view[idx] for view in self.views]

        # 应用转换（如果有）
        if self.transform:
            sample_views = [self.transform(view) for view in sample_views]

        if self.y is not None:
            return sample_views, self.y[idx]
        return sample_views