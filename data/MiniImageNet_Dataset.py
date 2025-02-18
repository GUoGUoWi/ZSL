from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image


class MiniImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        root_dir: 数据集的根目录
        split: 'train', 'val' 或 'test'
        transform: 应用于图像的预处理和增强
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.items = []

        # 假设标签文件是一个文本文件，每行格式为 "image_path label\n"
        with open(os.path.join(root_dir, f'{split}.txt'), 'r') as f:
            for line in f.readlines():
                image_path, label = line.strip().split()
                self.items.append((image_path, int(label)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, label = self.items[idx]
        image = Image.open(os.path.join(self.root_dir, image_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# 定义图像预处理步骤
data_transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_augmentation_transforms = transforms.Compose([
    transforms.RandomResizedCrop(84, scale=(0.8, 1.0)),  # 随机裁剪到84x84
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # 随机调整亮度、对比度和饱和度
    transforms.RandomRotation(10),  # 随机旋转±10度
    transforms.ToTensor(),  # 将PIL图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
])
# 创建数据集和数据加载器
root_dir = 'path/to/miniimagenet'


def get_mini_imgnet_dataloader(aug=True):
    if aug:
        transform = data_augmentation_transforms
    else:
        transform = data_transform
    train_dataset = MiniImageNetDataset(root_dir=root_dir, split='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = MiniImageNetDataset(root_dir=root_dir, split='test', transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    val_dataset = MiniImageNetDataset(root_dir=root_dir, split='val', transform=data_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    return train_loader, test_loader, val_loader

