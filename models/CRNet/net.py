# net.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ExpertModule, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, prototype_offsets):
        return F.relu(self.fc(prototype_offsets))


class CooperationModule(nn.Module):
    def __init__(self, num_experts, input_dim, output_dim):
        super(CooperationModule, self).__init__()
        self.output_dim = output_dim
        self.experts = nn.ModuleList([
            ExpertModule(input_dim, output_dim)
            for _ in range(num_experts)
        ])

    def forward(self, class_prototypes, cluster_centers):
        """
        Parameters:
        - class_prototypes: shape (num_classes, feature_dim) 或 (batch_size, feature_dim) 在训练时
        - cluster_centers: shape (num_clusters, feature_dim)
        """
        num_prototypes = class_prototypes.size(0)
        num_clusters = cluster_centers.size(0)
        feature_dim = class_prototypes.size(1)
        # 计算每个类原型与所有聚类中心的偏置
        expanded_prototypes = class_prototypes.unsqueeze(1).expand(-1, num_clusters, -1)  # [num_prototypes, num_clusters, feature_dim]
        expanded_centers = cluster_centers.unsqueeze(0).expand(num_prototypes, -1, -1)  # [num_prototypes, num_clusters, feature_dim]
        offsets = expanded_prototypes - expanded_centers  # [num_prototypes, num_clusters, feature_dim]
        offsets = offsets.view(-1, feature_dim)  # [num_prototypes * num_clusters, feature_dim]
        # 通过每个专家网络处理
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(offsets)  # [num_prototypes * num_clusters, output_dim]
            expert_output = expert_output.view(num_prototypes, num_clusters, -1)  # [num_prototypes, num_clusters, output_dim]
            expert_output = torch.sum(expert_output, dim=1)  # [num_prototypes, output_dim]
            expert_outputs.append(expert_output)
        # 合并所有专家的输出
        combined_output = sum(expert_outputs)  # [num_prototypes, output_dim]
        return combined_output


class RelationModule(nn.Module):
    def __init__(self, input_dim):
        super(RelationModule, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, input_dim)
        self.fc2 = nn.Linear(input_dim, 1)

    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=1)  # [batch_size, input_dim * 2]
        x = F.relu(self.fc1(combined))
        x = self.fc2(x)  # 移除 Sigmoid 激活
        return x


class CRNet(nn.Module):
    def __init__(self, visual_dim=64, semantic_dim=200, common_dim=256, num_experts=6):
        super(CRNet, self).__init__()
        self.common_dim = common_dim
        # 视觉特征映射到公共空间
        self.visual_mapper = nn.Sequential(
            nn.Linear(visual_dim, common_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(common_dim, common_dim)
        )
        # 语义特征映射到公共空间
        self.semantic_mapper = nn.Sequential(
            nn.Linear(semantic_dim, common_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(common_dim, common_dim)
        )
        # 专家模块和关系模块
        self.cooperation = CooperationModule(num_experts, common_dim, common_dim)
        self.relation = RelationModule(common_dim)

    def forward(self, visual_features, class_prototypes, cluster_centers=None, return_embeddings=False):
        """
        Args:
            visual_features: [batch_size, visual_dim] 视觉特征
            class_prototypes: [num_classes, semantic_dim] 或 [batch_size, semantic_dim] 在训练时
            cluster_centers: [num_clusters, feature_dim]
        """
        # 映射到公共空间
        visual_common = self.visual_mapper(visual_features)  # [batch_size, common_dim]
        semantic_common = self.semantic_mapper(class_prototypes)  # [num_classes, common_dim] 或 [batch_size, common_dim]

        if cluster_centers is not None:
            cluster_common = self.visual_mapper(cluster_centers)  # [num_clusters, common_dim]
            semantic_common = self.cooperation(semantic_common, cluster_common)  # [num_classes, common_dim] 或 [batch_size, common_dim]

        # 判断是训练模式（每对对应一个类原型）还是推理模式（所有类原型）
        if semantic_common.size(0) == visual_common.size(0):
            # 训练模式，每个样本对应一个类原型
            relations = self.relation(visual_common, semantic_common)  # [batch_size, 1]
            if return_embeddings:
                return relations, visual_common, semantic_common
            return relations
        else:
            # 推理模式，处理多类别关系
            num_classes = semantic_common.size(0)
            visual_common_exp = visual_common.unsqueeze(1).repeat(1, num_classes, 1)  # [batch_size, num_classes, common_dim]
            semantic_common_exp = semantic_common.unsqueeze(0).repeat(visual_common.size(0), 1, 1)  # [batch_size, num_classes, common_dim]
            # 计算样本与所有类别的相似度
            relations = self.relation(
                visual_common_exp.view(-1, self.common_dim),
                semantic_common_exp.view(-1, self.common_dim)
            )  # [batch_size * num_classes, 1]
            relations = relations.view(visual_common.size(0), num_classes)  # [batch_size, num_classes]
            if return_embeddings:
                return relations, visual_common, semantic_common
            return relations