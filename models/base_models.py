import torch
from torch import nn
from torch.nn.functional import normalize
from torchvision.models import resnet18
import torch.nn.functional as F

import clip

global TORCH_BACKEND_DEVICE  # torch 后端设备，可能是cpu、cuda、mps
if torch.cuda.is_available():
    TORCH_BACKEND_DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    TORCH_BACKEND_DEVICE = 'mps'
else:
    TORCH_BACKEND_DEVICE = 'cpu'


class MLP3(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=64):
        super(MLP3, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            # nn.Linear(32, output_dim),
        )
        self.twoCls_head = nn.Linear(output_dim, 2)
        self.threeCls_head = nn.Linear(output_dim, 3)
        self.fourCls_head = nn.Linear(output_dim, 4)
        self.fiveCls_head = nn.Linear(output_dim, 5)
        self.sixCls_head = nn.Linear(output_dim, 6)
        self.Regression_head = nn.Linear(output_dim, 1)
        self.finetune_head = nn.Linear(output_dim, 6)

    # @nan_detector
    def forward(self, x, mode='meta'):
        features = self.backbone(x)
        if mode == 'BinaryCls':
            return self.twoCls_head(features)
        elif mode == 'ThreeCls':
            return self.threeCls_head(features)
        elif mode == 'FourCls':
            return self.fourCls_head(features)
        elif mode == 'FiveCls':
            return self.fiveCls_head(features)
        elif mode == 'SixCls':
            return self.sixCls_head(features)
        elif mode == 'client':
            return self.finetune_head(features)
        elif mode == 'fet':
            return features
        else:
            return self.Regression_head(features)


class MultiViewResNet(nn.Module):
    def __init__(self, feature_dim=128, pretrained=False, num_features=1, batch_size=64):
        super(MultiViewResNet, self).__init__()
        self.batch_size = batch_size
        # 加载预训练的ResNet18作为基础特征提取器
        resnet = resnet18(pretrained=pretrained)
        # 移除最后的全连接层
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # 特征维度转换层（可选，用于调整特征维度）
        self.feature_projection = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU()
        ) if feature_dim != 512 else nn.Identity()

        # 保持与原MLP3相同的分类头
        self.twoCls_head = nn.Linear(feature_dim*num_features, 2)
        self.threeCls_head = nn.Linear(feature_dim*num_features, 3)
        self.fourCls_head = nn.Linear(feature_dim*num_features, 4)
        self.fiveCls_head = nn.Linear(feature_dim*num_features, 5)
        self.sixCls_head = nn.Linear(feature_dim*num_features, 6)
        self.Regression_head = nn.Linear(feature_dim*num_features, 1)
        self.finetune_head = nn.Linear(feature_dim*num_features, 10)

    def extract_single_view(self, x):
        """提取单个视角的特征，支持分批处理"""
        # 获取总样本数
        total_samples = x.size(0)
        features_list = []

        for start in range(0, total_samples, self.batch_size):
            end = min(start + self.batch_size, total_samples)
            batch_x = x[start:end]

            # [B, C, H, W] -> [B, 512, 1, 1]
            batch_features = self.backbone(batch_x)
            # [B, 512, 1, 1] -> [B, 512]
            batch_features = batch_features.view(batch_features.size(0), -1)
            # 投影到指定维度（如果需要）
            batch_features = self.feature_projection(batch_features)

            features_list.append(batch_features)

        # 拼接所有批次的特征
        features = torch.cat(features_list, dim=0)
        return features

    def extract_multi_view(self, view_list):
        """
        提取多视角特征并拼接，支持分批处理
        :param view_list: 视角图像张量列表，每个元素shape为[B, C, H, W]
        :return: 拼接后的特征张量 [B, feature_dim * num_views]
        """
        all_features = []
        for view_tensor in view_list:
            view_features = self.extract_single_view(view_tensor)
            all_features.append(view_features)

        # 在特征维度上拼接
        concatenated_features = torch.cat(all_features, dim=1)
        return concatenated_features

    def forward(self, x, mode='meta'):
        """
        前向传播
        :param x: 单视角输入为张量[B, C, H, W]，多视角输入为张量列表[tensor1, tensor2, ...]
        :param mode: 运行模式
        :return: 根据mode返回相应的输出
        """
        # 处理输入，获取特征
        if isinstance(x, list):
            # 多视角输入
            features = self.extract_multi_view(x)
        else:
            # 单视角输入
            features = self.extract_single_view(x)

        # 根据mode选择不同的输出头
        if mode == 'BinaryCls':
            return self.twoCls_head(features)
        elif mode == 'ThreeCls':
            return self.threeCls_head(features)
        elif mode == 'FourCls':
            return self.fourCls_head(features)
        elif mode == 'FiveCls':
            return self.fiveCls_head(features)
        elif mode == 'SixCls':
            return self.sixCls_head(features)
        elif mode == 'client':
            return self.finetune_head(features)
        elif mode == 'fet':
            return features
        else:
            return self.Regression_head(features)


class MultiViewCLIP_old(nn.Module):
    def __init__(self, feature_dim=128, clip_model_name="ViT-B/32", num_features=4, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.device = TORCH_BACKEND_DEVICE

        # 加载CLIP模型
        self.clip_model, _ = clip.load(clip_model_name, device=self.device)
        self.clip_model.eval()

        # CLIP的特征维度（对于ViT-B/32是512）
        clip_feature_dim = 512

        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(clip_feature_dim, feature_dim),
            nn.ReLU()
        ) if feature_dim != clip_feature_dim else nn.Identity()

        # 位置编码（可选使用）
        self.position_embeddings = nn.Parameter(torch.randn(4, feature_dim)).to(self.device)

        output_dim = feature_dim # feature_dim * num_features
        # 分类头
        self.twoCls_head = nn.Linear(output_dim, 2).half()
        self.threeCls_head = nn.Linear(output_dim, 3).half()
        self.fourCls_head = nn.Linear(output_dim, 4).half()
        self.fiveCls_head = nn.Linear(output_dim, 5).half()
        self.sixCls_head = nn.Linear(output_dim, 6).half()
        self.Regression_head = nn.Linear(output_dim, 1).half()
        self.finetune_head = nn.Linear(output_dim, 10).half()

    def extract_single_view(self, x):
        """
        提取单个视角的特征，支持分批处理

        Args:
            x: 输入张量 [B, C, H, W]
        Returns:
            features: 特征张量 [B, feature_dim]
        """
        total_samples = x.size(0)
        features_list = []

        for start in range(0, total_samples, self.batch_size):
            end = min(start + self.batch_size, total_samples)
            batch_x = x[start:end].to(self.device)

            # 确保输入范围在[-1, 1]之间
            if batch_x.min() >= 0 and batch_x.max() <= 1:
                batch_x = 2 * batch_x - 1

            with torch.no_grad():
                # 提取CLIP特征
                batch_features = self.clip_model.encode_image(batch_x)
                batch_features = normalize(batch_features, dim=-1)

            # 投影到指定维度
            batch_features = self.feature_projection(batch_features)
            features_list.append(batch_features.cpu())  # 移到CPU以节省GPU内存

        # 拼接所有批次的特征
        features = torch.cat(features_list, dim=0)
        return features

    def extract_multi_view(self, view_list):
        """
        提取多视角特征并拼接

        Args:
            view_list: 视角图像张量列表，每个元素shape为[B, C, H, W]
        Returns:
            concatenated_features: 拼接后的特征张量 [B, feature_dim * num_views]
        """
        all_features = []
        for view_tensor in view_list:
            view_features = self.extract_single_view(view_tensor)
            all_features.append(view_features)

        # 在特征维度上拼接
        concatenated_features = torch.cat(all_features, dim=1)
        return concatenated_features

    def extract_features_without_position(self, image_parts):
        """
        在不知道位置信息的情况下提取特征

        Args:
            image_parts: 图像块列表 [tensor1, tensor2, ...]
        Returns:
            features: 融合后的特征 [B, feature_dim]
        """
        total_samples = image_parts[0].size(0)
        all_features = []

        # 处理每个图像块
        for img_part in image_parts:
            features_list = []

            for start in range(0, total_samples, self.batch_size):
                end = min(start + self.batch_size, total_samples)
                batch_x = img_part[start:end].to(self.device)

                # 调整图像大小至224x224
                batch_x = F.interpolate(batch_x, size=(224, 224), mode='bilinear', align_corners=False)

                # 确保值范围在[-1, 1]之间
                if batch_x.min() >= 0 and batch_x.max() <= 1:
                    batch_x = 2 * batch_x - 1

                with torch.no_grad():
                    # 提取CLIP特征
                    batch_features = self.clip_model.encode_image(batch_x)
                    batch_features = normalize(batch_features, dim=-1)

                # 投影到指定维度
                batch_features = self.feature_projection(batch_features)
                features_list.append(batch_features.cpu())

            # 合并当前部分的所有批次特征
            part_features = torch.cat(features_list, dim=0)
            all_features.append(part_features)

        # 堆叠所有部分的特征并取平均
        stacked_features = torch.stack(all_features, dim=1)  # [B, num_parts, feature_dim]
        averaged_features = torch.mean(stacked_features, dim=1)  # [B, feature_dim]

        return averaged_features

    def extract_features_with_position(self, image_parts, positions):
        """
        在知道位置信息的情况下提取特征

        Args:
            image_parts: 图像块列表 [tensor1, tensor2, ...]
            positions: 对应的位置列表 [0,1,2,3]表示左上、右上、左下、右下
        Returns:
            features: 融合后的特征 [B, feature_dim]
        """
        total_samples = image_parts[0].size(0)
        all_features = []

        # 处理每个图像块
        for img_part, pos in zip(image_parts, positions):
            features_list = []

            for start in range(0, total_samples, self.batch_size):
                end = min(start + self.batch_size, total_samples)
                batch_x = img_part[start:end].to(self.device)

                # 调整图像大小至224x224
                batch_x = F.interpolate(batch_x, size=(224, 224), mode='bilinear', align_corners=False)

                # 确保值范围在[-1, 1]之间
                if batch_x.min() >= 0 and batch_x.max() <= 1:
                    batch_x = 2 * batch_x - 1

                with torch.no_grad():
                    # 提取CLIP特征
                    batch_features = self.clip_model.encode_image(batch_x)
                    batch_features = normalize(batch_features, dim=-1)

                # 投影到指定维度
                batch_features = self.feature_projection(batch_features)

                # 添加位置编码
                batch_features = batch_features + self.position_embeddings[pos]

                features_list.append(batch_features.cpu())

            # 合并当前部分的所有批次特征
            part_features = torch.cat(features_list, dim=0)
            all_features.append(part_features)

        # 堆叠所有部分的特征并取平均
        stacked_features = torch.stack(all_features, dim=1)
        averaged_features = torch.mean(stacked_features, dim=1)

        return averaged_features

    def forward(self, x, mode='meta', positions=None):
        """
        扩展的前向传播函数

        Args:
            x: 可以是：
               - 单视角输入张量 [B, C, H, W]
               - 多视角输入张量列表 [tensor1, tensor2, ...]
               - 部分图像输入列表（当mode='partial'时）
            mode: 运行模式
            positions: 可选的位置信息
        Returns:
            根据mode返回相应的输出
        """
        if mode == 'partial' or mode == 'fet' or mode == 'client':
            if positions is not None:
                features = self.extract_features_with_position(x, positions)
            else:
                features = self.extract_features_without_position(x)
        elif isinstance(x, list):
            features = self.extract_multi_view(x)
        else:
            features = self.extract_single_view(x)

        features = features.to(self.device) # .float()

        # 根据mode选择不同的输出头
        if mode == 'BinaryCls':
            return self.twoCls_head(features)
        elif mode == 'ThreeCls':
            return self.threeCls_head(features)
        elif mode == 'FourCls':
            return self.fourCls_head(features)
        elif mode == 'FiveCls':
            return self.fiveCls_head(features)
        elif mode == 'SixCls':
            return self.sixCls_head(features)
        elif mode == 'client':
            return self.finetune_head(features)
        elif mode == 'fet' or mode == 'partial':
            return features
        else:
            return self.Regression_head(features)


class MultiViewCLIP(nn.Module):
    def __init__(self, feature_dim=512, clip_model_name="ViT-B/32", num_features=4, batch_size=64, allow_backbone=False):
        super().__init__()
        self.batch_size = batch_size
        self.device = TORCH_BACKEND_DEVICE

        # 设置默认精度
        self.dtype = torch.float32  # 或 torch.float32

        # 加载CLIP模型
        self.clip_model, _ = clip.load(clip_model_name, device=self.device)
        if not allow_backbone:
            self.clip_model.eval().to(dtype=self.dtype)
        else:
            self.clip_model.train().to(dtype=self.dtype)

        clip_feature_dim = 512
        # 统一精度
        # self.feature_projection = nn.Sequential(
        #     nn.Linear(clip_feature_dim, feature_dim),
        #     nn.ReLU()
        # ).to(dtype=self.dtype)

        # 位置编码也使用相同精度
        self.position_embeddings = nn.Parameter(
            torch.randn(4, feature_dim, dtype=self.dtype)
        ).to(self.device)

        output_dim = feature_dim  # feature_dim * num_features
        # 分类头使用相同精度
        # 分类头
        self.twoCls_head = nn.Linear(output_dim, 2).to(dtype=self.dtype)
        self.threeCls_head = nn.Linear(output_dim, 3).to(dtype=self.dtype)
        self.fourCls_head = nn.Linear(output_dim, 4).to(dtype=self.dtype)
        self.fiveCls_head = nn.Linear(output_dim, 5).to(dtype=self.dtype)
        self.sixCls_head = nn.Linear(output_dim, 6).to(dtype=self.dtype)
        self.Regression_head = nn.Linear(output_dim, 1).to(dtype=self.dtype)
        self.finetune_head = nn.Linear(output_dim, 10).to(dtype=self.dtype)

    def extract_single_view(self, x):
        total_samples = x.size(0)
        features_list = []

        for start in range(0, total_samples, self.batch_size):
            end = min(start + self.batch_size, total_samples)
            batch_x = x[start:end].to(self.device)

            # 确保输入范围和精度
            if batch_x.min() >= 0 and batch_x.max() <= 1:
                batch_x = (2 * batch_x - 1).to(dtype=self.dtype)

            with torch.no_grad():
                # 提取CLIP特征并确保精度一致
                batch_features = self.clip_model.encode_image(batch_x)
                batch_features = normalize(batch_features, dim=-1)
                batch_features = batch_features.to(dtype=self.dtype)

            # 投影
            # batch_features = self.feature_projection(batch_features)
            features_list.append(batch_features.cpu())

        features = torch.cat(features_list, dim=0)
        return features

    def extract_multi_view(self, view_list):
        """
        提取多视角特征并拼接

        Args:
            view_list: 视角图像张量列表，每个元素shape为[B, C, H, W]
        Returns:
            concatenated_features: 拼接后的特征张量 [B, feature_dim * num_views]
        """
        all_features = []
        for view_tensor in view_list:
            view_features = self.extract_single_view(view_tensor)
            all_features.append(view_features)

        # 在特征维度上拼接
        concatenated_features = torch.cat(all_features, dim=1)
        return concatenated_features

    def extract_features_without_position(self, image_parts):
        """
        在不知道位置信息的情况下提取特征

        Args:
            image_parts: 图像块列表 [tensor1, tensor2, ...]
        Returns:
            features: 融合后的特征 [B, feature_dim]
        """
        total_samples = image_parts[0].size(0)
        all_features = []

        # 处理每个图像块
        for img_part in image_parts:
            features_list = []

            for start in range(0, total_samples, self.batch_size):
                end = min(start + self.batch_size, total_samples)
                batch_x = img_part[start:end].to(self.device)

                # 调整图像大小至224x224
                batch_x = F.interpolate(batch_x, size=(224, 224), mode='bilinear', align_corners=False)

                # 确保值范围在[-1, 1]之间
                if batch_x.min() >= 0 and batch_x.max() <= 1:
                    batch_x = 2 * batch_x - 1

                with torch.no_grad():
                    # 提取CLIP特征
                    batch_features = self.clip_model.encode_image(batch_x)
                    batch_features = normalize(batch_features, dim=-1)

                # 投影到指定维度
                # batch_features = self.feature_projection(batch_features)
                features_list.append(batch_features.cpu())

            # 合并当前部分的所有批次特征
            part_features = torch.cat(features_list, dim=0)
            all_features.append(part_features)

        # 堆叠所有部分的特征并取平均
        stacked_features = torch.stack(all_features, dim=1)  # [B, num_parts, feature_dim]
        averaged_features = torch.mean(stacked_features, dim=1)  # [B, feature_dim]

        return averaged_features

    def extract_features_with_position(self, image_parts, positions):
        """
        在知道位置信息的情况下提取特征

        Args:
            image_parts: 图像块列表 [tensor1, tensor2, ...]
            positions: 对应的位置列表 [0,1,2,3]表示左上、右上、左下、右下
        Returns:
            features: 融合后的特征 [B, feature_dim]
        """
        total_samples = image_parts[0].size(0)
        all_features = []

        # 处理每个图像块
        for img_part, pos in zip(image_parts, positions):
            features_list = []

            for start in range(0, total_samples, self.batch_size):
                end = min(start + self.batch_size, total_samples)
                batch_x = img_part[start:end].to(self.device)

                # 调整图像大小至224x224
                batch_x = F.interpolate(batch_x, size=(224, 224), mode='bilinear', align_corners=False)

                # 确保值范围在[-1, 1]之间
                if batch_x.min() >= 0 and batch_x.max() <= 1:
                    batch_x = 2 * batch_x - 1

                with torch.no_grad():
                    # 提取CLIP特征
                    batch_features = self.clip_model.encode_image(batch_x)
                    batch_features = normalize(batch_features, dim=-1)

                # 投影到指定维度
                # batch_features = self.feature_projection(batch_features)

                # 添加位置编码
                batch_features = batch_features + self.position_embeddings[pos]

                features_list.append(batch_features.cpu())

            # 合并当前部分的所有批次特征
            part_features = torch.cat(features_list, dim=0)
            all_features.append(part_features)

        # 堆叠所有部分的特征并取平均
        stacked_features = torch.stack(all_features, dim=1)
        averaged_features = torch.mean(stacked_features, dim=1)

        return averaged_features

    def forward(self, x, mode='meta', positions=None):
        """
                扩展的前向传播函数

                Args:
                    x: 可以是：
                       - 单视角输入张量 [B, C, H, W]
                       - 多视角输入张量列表 [tensor1, tensor2, ...]
                       - 部分图像输入列表（当mode='partial'时）
                    mode: 运行模式
                    positions: 可选的位置信息
                Returns:
                    根据mode返回相应的输出
                """
        if mode == 'partial' or mode == 'fet' or mode == 'client':
            if positions is not None:
                features = self.extract_features_with_position(x, positions)
            else:
                features = self.extract_features_without_position(x)
        elif isinstance(x, list):
            features = self.extract_multi_view(x)
        else:
            features = self.extract_single_view(x)

        features = features.to(self.device)  # .float()

        # 确保特征精度一致
        features = features.to(self.device, dtype=self.dtype)

        if mode == 'client':
            # 添加梯度检查
            if torch.isnan(features).any():
                print("NaN detected in features before finetune_head")
                features = torch.nan_to_num(features, nan=0.0)

            output = self.finetune_head(features)

            if torch.isnan(output).any():
                print("NaN detected in finetune_head output")
                print(f"Features stats: min={features.min()}, max={features.max()}, mean={features.mean()}")
                print(f"Output stats: min={output.min()}, max={output.max()}, mean={output.mean()}")

            return output
        elif mode == 'BinaryCls':
            return self.twoCls_head(features)
        elif mode == 'ThreeCls':
            return self.threeCls_head(features)
        elif mode == 'FourCls':
            return self.fourCls_head(features)
        elif mode == 'FiveCls':
            return self.fiveCls_head(features)
        elif mode == 'SixCls':
            return self.sixCls_head(features)
        elif mode == 'fet' or mode == 'partial':
            return features
        else:
            return self.Regression_head(features)
