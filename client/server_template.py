import logging

import numpy as np
import torch
from torch import nn, optim
from torch.nn.functional import normalize
from torch.utils.data import TensorDataset, DataLoader
from models.CRNet.models_cls import CRnetCls
from models.CRNet.train_CRNet_cls import test_CRNet
from gensim.models import KeyedVectors
from tqdm import tqdm
import clip

from models.CRNet.trainer import generate_label_matrix_clip

THERMAL_COMFORT_DESCRIPTIONS_SEEN = {
    0: "very cold uncomfortable thermal condition",
    1: "cold slightly uncomfortable thermal environment",
    2: "slightly cool but acceptable thermal state",
    3: "neutral comfortable thermal condition",
    4: "slightly warm but acceptable thermal state",
    5: "hot uncomfortable thermal environment"
}

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

CIFAR10_DESCRIPTIONS_DETAILED = {
    0: 'an airplane, a fixed-wing aircraft',
    1: 'an automobile, a passenger car',
    2: 'a bird, a small flying animal',
    3: 'a cat, a domestic feline pet',
    4: 'a deer, a wild forest animal with antlers',
    5: 'a dog, a domestic canine pet',
    6: 'a frog, an amphibian with smooth skin',
    7: 'a horse, a large four-legged mammal',
    8: 'a ship, a large watercraft vessel',
    9: 'a truck, a large road vehicle for carrying cargo'
}

def generate_label_matrix(labels, glove_model_name="./data/glove-wiki-gigaword-200.txt"):
    """
    生成标签词向量矩阵
    """
    glove_vectors = KeyedVectors.load_word2vec_format(glove_model_name)
    label_vectors = np.zeros((len(labels), glove_vectors.vector_size))
    for i, label in enumerate(labels):
        word_vectors = [glove_vectors[word] for word in label.split() if word in glove_vectors]
        if word_vectors:
            label_vectors[i] = np.mean(word_vectors, axis=0)
    return label_vectors


global TORCH_BACKEND_DEVICE  # torch 后端设备，可能是cpu、cuda、mps
if torch.cuda.is_available():
    TORCH_BACKEND_DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    TORCH_BACKEND_DEVICE = torch.device('mps')
else:
    TORCH_BACKEND_DEVICE = torch.device("cpu")


class VFLServer:
    def __init__(self, input_dim, aligned_test_features, aligned_test_labels, aligned_train_features, aligned_train_labels, unaligned_features, unaligned_labels, device=None, num_labels=6,
                 num_client=3,except_classes=[1,2,5], dataset='comfort', clip=False):
        self.batch_size = 32
        self.except_classe = except_classes
        self.device = TORCH_BACKEND_DEVICE
        self.cls_criterion = nn.CrossEntropyLoss().to(self.device)
        # self.aligned_test_feature = torch.cat(aligned_test_features, dim=1).to(self.device)
        self.aligned_test_features = aligned_test_features
            # torch.cat(aligned_features, dim=1).to(self.device))
        self.aligned_test_labels = aligned_test_labels.to(self.device)
        self.aligned_train_feature = torch.cat(aligned_train_features, dim=1).to(self.device)
            # torch.cat(aligned_features, dim=1).to(self.device))
        self.aligned_train_labels = aligned_train_labels.to(self.device)
        self.unaligned_features = torch.from_numpy(unaligned_features['features']).to(torch.float32).to(self.device)
        self.unaligned_labels = unaligned_labels.to(self.device)
        self.num_labels = num_labels
        # self.CRNet = CRnetCls(num_experts=3, semantic_dim=200, visual_dim=16 * num_client, hidden_dim=100).to(self.device)

        self.dataset = dataset
        # 获取类别的语义嵌入
        seen_classes = list(range(num_labels))  # 假设前6类为seen
        self.seen_classes = seen_classes
        if dataset == 'comfort':
            DESCRIPTIONS = THERMAL_COMFORT_DESCRIPTIONS_SEEN
        elif dataset == 'modelnet':
            DESCRIPTIONS = MODELNET_DESCRIPTIONS_SEEN
        elif dataset == 'fmnist':
            if clip:
                DESCRIPTIONS = CIFAR10_DESCRIPTIONS_DETAILED
            else:
                DESCRIPTIONS = FMNIST_DESCRIPTIONS_SEEN
        else:
            raise RuntimeError('NO SUCH DATASET')

        descriptions = [DESCRIPTIONS[i] for i in self.seen_classes]
        if clip:
            self.seen_semantic_prototypes = generate_label_matrix_clip(descriptions)
        else:
            self.seen_semantic_prototypes = generate_label_matrix(descriptions,
                                                                  glove_model_name="./data/glove-wiki-gigaword-200.txt")
        self.seen_semantic_prototypes = torch.from_numpy(self.seen_semantic_prototypes).float().to(device)
        self.w1 = nn.Parameter(torch.ones(num_labels) / num_labels)
        self.w2 = nn.Parameter(torch.ones(num_labels) / num_labels)
        self.w3 = nn.Parameter(torch.ones(num_labels) / num_labels)
        self.w4 = nn.Parameter(torch.ones(num_labels) / num_labels)
        self.w_criterion = nn.CrossEntropyLoss()

    # def aggregate_models(self, local_models, client_weights=None):
    #     """
    #     聚合本地模型到全局模型
    #
    #     参数:
    #     global_model: 全局模型 (PyTorch 模型)
    #     local_models: 本地模型参数列表 (List[Dict])
    #     client_weights: 每个客户端的权重 (List[float])
    #
    #     返回:
    #     None, 直接在 global_model 中更新参数
    #     """
    #     if client_weights is None:
    #         client_weights = [1.0/len(local_models)] * len(local_models)
    #     global_model = self.CRNet
    #     # 初始化全局模型参数
    #     global_state_dict = global_model.state_dict()
    #     # 对每个参数进行加权平均
    #     for key in global_state_dict.keys():
    #         global_state_dict[key] = sum(
    #             client_weights[i] * local_models[i].state_dict()[key].clone() for i in range(len(local_models))) / sum(client_weights)
    #
    #     # 更新全局模型参数
    #     global_model.load_state_dict(global_state_dict)

    # def get_CRNet_model(self):
    #     return self.CRNet

    def calculate_mutual_information(self, x, y, n_bins=50):
        """
        计算两个特征之间的互信息
        x, y: torch.Tensor, 特征向量
        """
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        # 将特征转换为离散值
        x_bins = np.histogram_bin_edges(x, bins=n_bins)
        y_bins = np.histogram_bin_edges(y, bins=n_bins)

        x_discrete = np.digitize(x, x_bins)
        y_discrete = np.digitize(y, y_bins)

        # 计算联合概率和边缘概率
        xy_hist = np.histogram2d(x_discrete, y_discrete, bins=[n_bins, n_bins])[0]
        p_xy = xy_hist / xy_hist.sum()
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)

        # 计算互信息
        mi = 0
        for i in range(n_bins):
            for j in range(n_bins):
                if p_xy[i, j] > 0:
                    mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))

        return mi

    def CRNet_progress(self, client_models, client_cluster_centers, client_class_prototypes, client_gen_datas, client_gen_labels):
        test_features = self.aligned_test_features
        test_labels = self.aligned_test_labels
        print(test_labels.shape)
        print(test_features[0].shape)
        weights = [self.w1, self.w2, self.w3]
        test_CRNet(client_models, client_cluster_centers, test_features, test_labels, weights, device=self.device, dataset=self.dataset)
        # test_CRNet(client_models, client_cluster_centers, test_features, test_labels, device=self.device)
        # augmented_data_list = self.tri_training(client_models, client_cluster_centers, client_datas)
        num_epochs = 10
        batch_size = 32
        client_num = len(client_models)
        # 依次训练每个客户端对应的权重
        for client_idx in range(client_num):
            print(f"\nTraining weights for Client {client_idx + 1}")
            # 只优化当前客户端对应的权重
            current_weight = weights[client_idx]
            optimizer = optim.Adam([
                {'params': client_models[client_idx].parameters()},
                {'params': current_weight}
            ], lr=0.001)
            criterion = nn.CrossEntropyLoss()
            # 准备当前客户端的数据
            features = torch.from_numpy(client_gen_datas[client_idx]).float().to(self.device)
            labels = client_gen_labels
            # 创建数据加载器
            dataset = TensorDataset(features[:], labels[:])
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for epoch in tqdm(range(num_epochs)):
                total_loss = 0
                correct = 0
                total = 0
                for batch_features, batch_labels in dataloader:
                    optimizer.zero_grad()
                    # 获取所有模型的预测和中间特征
                    model_outputs = []
                    intermediate_visual_features = []
                    intermediate_semantic_features = []
                    for i, model in enumerate(client_models):
                        if client_idx == i:
                            model.train()
                            outputs, visual_common, semantic_common = model(
                                batch_features,
                                client_class_prototypes[i],
                                client_cluster_centers[i],
                                return_embeddings=True)
                        else:
                            model.eval()
                            with torch.no_grad():
                                outputs, visual_common, semantic_common = model(
                                    batch_features,
                                    client_class_prototypes[i],
                                    client_cluster_centers[i],
                                    return_embeddings=True)
                        model_outputs.append(outputs.cpu())
                        intermediate_visual_features.append(visual_common)
                        intermediate_semantic_features.append(semantic_common)

                    # 计算模型对之间的互信息权重
                    mi_matrix = np.zeros((client_num, client_num))
                    for i in range(client_num):
                        for j in range(client_num):
                            mi_visual = self.calculate_mutual_information(
                                intermediate_visual_features[i].mean(dim=0),
                                intermediate_visual_features[j].mean(dim=0)
                            )
                            mi_semantic = self.calculate_mutual_information(
                                intermediate_semantic_features[i].mean(dim=0),
                                intermediate_semantic_features[j].mean(dim=0 )
                            )
                            mi_matrix[i, j] = (mi_visual + mi_semantic) / 2

                    # 计算集成预测
                    final_predictions = torch.zeros(batch_features.size(0), self.num_labels)

                    if client_idx == 0:
                        # 1-2对
                        pair_pred = (model_outputs[0] * current_weight.unsqueeze(0) +
                                     model_outputs[1] * weights[1].detach().unsqueeze(0))
                        final_predictions += mi_matrix[0, 1] * pair_pred
                        # 1-3对
                        pair_pred = (model_outputs[0] * current_weight.unsqueeze(0) +
                                     model_outputs[2] * weights[2].detach().unsqueeze(0))
                        final_predictions += mi_matrix[0, 2] * pair_pred
                        # 2-3对
                        pair_pred = (model_outputs[1] * weights[1].detach().unsqueeze(0) +
                                     model_outputs[2] * weights[2].detach().unsqueeze(0))
                        final_predictions += mi_matrix[1, 2] * pair_pred



                    elif client_idx == 1:
                        # 1-2对
                        pair_pred = (model_outputs[0] * weights[0].detach().unsqueeze(0) +
                                     model_outputs[1] * current_weight.unsqueeze(0))
                        final_predictions += mi_matrix[0, 1] * pair_pred
                        # 1-3对
                        pair_pred = (model_outputs[0] * weights[0].detach().unsqueeze(0) +
                                     model_outputs[2] * weights[2].detach().unsqueeze(0))
                        final_predictions += mi_matrix[0, 2] * pair_pred

                        # 2-3对
                        pair_pred = (model_outputs[1] * current_weight.unsqueeze(0) +
                                     model_outputs[2] * weights[2].detach().unsqueeze(0))
                        final_predictions += mi_matrix[1, 2] * pair_pred



                    elif client_idx == 2:
                        # 1-2对
                        pair_pred = (model_outputs[0] * weights[0].detach().unsqueeze(0) +
                                     model_outputs[1] * weights[1].detach().unsqueeze(0))
                        final_predictions += mi_matrix[0, 1] * pair_pred
                        # 1-3对
                        pair_pred = (model_outputs[0] * weights[0].detach().unsqueeze(0) +
                                     model_outputs[2] * current_weight.unsqueeze(0))
                        final_predictions += mi_matrix[0, 2] * pair_pred

                        # 2-3对
                        pair_pred = (model_outputs[1] * weights[1].detach().unsqueeze(0) +
                                     model_outputs[2] * current_weight.unsqueeze(0))
                        final_predictions += mi_matrix[1, 2] * pair_pred


                    elif client_idx == 3:
                        # 1-2对
                        pair_pred = (model_outputs[0] * weights[0].detach().unsqueeze(0) +
                                     model_outputs[1] * weights[1].detach().unsqueeze(0))
                        final_predictions += mi_matrix[0, 1] * pair_pred
                        # 1-3对
                        pair_pred = (model_outputs[0] * weights[0].detach().unsqueeze(0) +
                                     model_outputs[2] * weights[2].detach().unsqueeze(0))
                        final_predictions += mi_matrix[0, 2] * pair_pred
                        # 2-3对
                        pair_pred = (model_outputs[1] * weights[1].detach().unsqueeze(0) +
                                     model_outputs[2] * weights[2].detach().unsqueeze(0))
                        final_predictions += mi_matrix[1, 2] * pair_pred

                    # 计算损失并更新权重
                    loss = criterion(final_predictions, batch_labels)
                    loss.backward()
                    optimizer.step()
                    # 计算准确率
                    _, predicted = torch.max(final_predictions, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                    total_loss += loss.item()

                epoch_acc = 100 * correct / total
                epoch_loss = total_loss / len(dataloader)
                print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
                print(f'Current weight vector: {current_weight.data}')

            # 在训练完成后更新原始权重
            with torch.no_grad():
                weights[client_idx].data = current_weight.data.clone()

        test_CRNet(client_models, client_cluster_centers, test_features, test_labels, weights, device=self.device, dataset=self.dataset)
        return client_models, client_cluster_centers

    def CRNet_progress_img(self, client_models, client_cluster_centers, client_class_prototypes, client_gen_datas, client_gen_labels, clip=False):
        test_features = self.aligned_test_features
        test_labels = self.aligned_test_labels
        print(test_labels.shape)
        print(test_features[0].shape)
        weights = [self.w1, self.w2, self.w3, self.w4]
        test_CRNet(client_models, client_cluster_centers, test_features, test_labels, weights, device=self.device,
                   dataset=self.dataset, clip=clip)
        # test_CRNet(client_models, client_cluster_centers, test_features, test_labels, device=self.device)
        # augmented_data_list = self.tri_training(client_models, client_cluster_centers, client_datas)
        num_epochs = 10
        batch_size = 32
        client_num = len(client_models)
        # 依次训练每个客户端对应的权重
        for client_idx in range(client_num):
            print(f"\nTraining weights for Client {client_idx + 1}")
            # 只优化当前客户端对应的权重
            current_weight = weights[client_idx]
            optimizer = optim.Adam([
                {'params': client_models[client_idx].parameters()},
                {'params': current_weight}
            ], lr=0.001)
            criterion = nn.CrossEntropyLoss()
            # 准备当前客户端的数据
            features = torch.from_numpy(client_gen_datas[client_idx]).float().to(self.device)
            labels = client_gen_labels
            # 创建数据加载器
            dataset = TensorDataset(features[:], labels[:])
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for epoch in tqdm(range(num_epochs)):
                total_loss = 0
                correct = 0
                total = 0
                for batch_features, batch_labels in dataloader:
                    optimizer.zero_grad()
                    # 获取所有模型的预测和中间特征
                    model_outputs = []
                    intermediate_visual_features = []
                    intermediate_semantic_features = []
                    for i, model in enumerate(client_models):
                        if client_idx == i:
                            model.train()
                            outputs, visual_common, semantic_common = model(
                                batch_features,
                                client_class_prototypes[i],
                                client_cluster_centers[i],
                                return_embeddings=True)
                        else:
                            model.eval()
                            with torch.no_grad():
                                outputs, visual_common, semantic_common = model(
                                    batch_features,
                                    client_class_prototypes[i],
                                    client_cluster_centers[i],
                                    return_embeddings=True)
                        model_outputs.append(outputs.cpu())
                        intermediate_visual_features.append(visual_common)
                        intermediate_semantic_features.append(semantic_common)

                    # 计算模型对之间的互信息权重
                    mi_matrix = np.zeros((client_num, client_num))
                    for i in range(client_num):
                        for j in range(client_num):
                            mi_visual = self.calculate_mutual_information(
                                intermediate_visual_features[i].mean(dim=0),
                                intermediate_visual_features[j].mean(dim=0)
                            )
                            mi_semantic = self.calculate_mutual_information(
                                intermediate_semantic_features[i].mean(dim=0),
                                intermediate_semantic_features[j].mean(dim=0 )
                            )
                            mi_matrix[i, j] = (mi_visual + mi_semantic) / 2

                    # 计算集成预测
                    final_predictions = torch.zeros(batch_features.size(0), self.num_labels)

                    if client_idx == 0:
                        # 1-2对
                        pair_pred = (model_outputs[0] * current_weight.unsqueeze(0) +
                                     model_outputs[1] * weights[1].detach().unsqueeze(0))
                        final_predictions += mi_matrix[0, 1] * pair_pred
                        # 1-3对
                        pair_pred = (model_outputs[0] * current_weight.unsqueeze(0) +
                                     model_outputs[2] * weights[2].detach().unsqueeze(0))
                        final_predictions += mi_matrix[0, 2] * pair_pred
                        # 1-4对
                        pair_pred = (model_outputs[0] * current_weight.unsqueeze(0) +
                                     model_outputs[3] * weights[3].detach().unsqueeze(0))
                        final_predictions += mi_matrix[0, 3] * pair_pred
                        # 2-3对
                        pair_pred = (model_outputs[1] * weights[1].detach().unsqueeze(0) +
                                     model_outputs[2] * weights[2].detach().unsqueeze(0))
                        final_predictions += mi_matrix[1, 2] * pair_pred
                        # 2-4对
                        pair_pred = (model_outputs[1] * weights[1].detach().unsqueeze(0) +
                                     model_outputs[3] * weights[3].detach().unsqueeze(0))
                        final_predictions += mi_matrix[1, 3] * pair_pred
                        # 3-4对
                        pair_pred = (model_outputs[2] * weights[2].detach().unsqueeze(0) +
                                     model_outputs[3] * weights[3].detach().unsqueeze(0))
                        final_predictions += mi_matrix[2, 3] * pair_pred

                    elif client_idx == 1:
                        # 1-2对
                        pair_pred = (model_outputs[0] * weights[0].detach().unsqueeze(0) +
                                     model_outputs[1] * current_weight.unsqueeze(0))
                        final_predictions += mi_matrix[0, 1] * pair_pred
                        # 1-3对
                        pair_pred = (model_outputs[0] * weights[0].detach().unsqueeze(0) +
                                     model_outputs[2] * weights[2].detach().unsqueeze(0))
                        final_predictions += mi_matrix[0, 2] * pair_pred
                        # 1-4对
                        pair_pred = (model_outputs[0] * weights[0].detach().unsqueeze(0) +
                                     model_outputs[3] * weights[3].detach().unsqueeze(0))
                        final_predictions += mi_matrix[0, 3] * pair_pred
                        # 2-3对
                        pair_pred = (model_outputs[1] * current_weight.unsqueeze(0) +
                                     model_outputs[2] * weights[2].detach().unsqueeze(0))
                        final_predictions += mi_matrix[1, 2] * pair_pred
                        # 2-4对
                        pair_pred = (model_outputs[1] * current_weight.unsqueeze(0) +
                                     model_outputs[3] * weights[3].detach().unsqueeze(0))
                        final_predictions += mi_matrix[1, 3] * pair_pred
                        # 3-4对
                        pair_pred = (model_outputs[2] * weights[2].detach().unsqueeze(0) +
                                     model_outputs[3] * weights[3].detach().unsqueeze(0))
                        final_predictions += mi_matrix[2, 3] * pair_pred

                    elif client_idx == 2:
                        # 1-2对
                        pair_pred = (model_outputs[0] * weights[0].detach().unsqueeze(0) +
                                     model_outputs[1] * weights[1].detach().unsqueeze(0))
                        final_predictions += mi_matrix[0, 1] * pair_pred
                        # 1-3对
                        pair_pred = (model_outputs[0] * weights[0].detach().unsqueeze(0) +
                                     model_outputs[2] * current_weight.unsqueeze(0))
                        final_predictions += mi_matrix[0, 2] * pair_pred
                        # 1-4对
                        pair_pred = (model_outputs[0] * weights[0].detach().unsqueeze(0) +
                                     model_outputs[3] * weights[3].detach().unsqueeze(0))
                        final_predictions += mi_matrix[0, 3] * pair_pred
                        # 2-3对
                        pair_pred = (model_outputs[1] * weights[1].detach().unsqueeze(0) +
                                     model_outputs[2] * current_weight.unsqueeze(0))
                        final_predictions += mi_matrix[1, 2] * pair_pred
                        # 2-4对
                        pair_pred = (model_outputs[1] * weights[1].detach().unsqueeze(0) +
                                     model_outputs[3] * weights[3].detach().unsqueeze(0))
                        final_predictions += mi_matrix[1, 3] * pair_pred
                        # 3-4对
                        pair_pred = (model_outputs[2] * current_weight.unsqueeze(0) +
                                     model_outputs[3] * weights[3].detach().unsqueeze(0))
                        final_predictions += mi_matrix[2, 3] * pair_pred

                    elif client_idx == 3:
                        # 1-2对
                        pair_pred = (model_outputs[0] * weights[0].detach().unsqueeze(0) +
                                     model_outputs[1] * weights[1].detach().unsqueeze(0))
                        final_predictions += mi_matrix[0, 1] * pair_pred
                        # 1-3对
                        pair_pred = (model_outputs[0] * weights[0].detach().unsqueeze(0) +
                                     model_outputs[2] * weights[2].detach().unsqueeze(0))
                        final_predictions += mi_matrix[0, 2] * pair_pred
                        # 1-4对
                        pair_pred = (model_outputs[0] * weights[0].detach().unsqueeze(0) +
                                     model_outputs[3] * current_weight.unsqueeze(0))
                        final_predictions += mi_matrix[0, 3] * pair_pred
                        # 2-3对
                        pair_pred = (model_outputs[1] * weights[1].detach().unsqueeze(0) +
                                     model_outputs[2] * weights[2].detach().unsqueeze(0))
                        final_predictions += mi_matrix[1, 2] * pair_pred
                        # 2-4对
                        pair_pred = (model_outputs[1] * weights[1].detach().unsqueeze(0) +
                                     model_outputs[3] * current_weight.unsqueeze(0))
                        final_predictions += mi_matrix[1, 3] * pair_pred
                        # 3-4对
                        pair_pred = (model_outputs[2] * weights[2].detach().unsqueeze(0) +
                                     model_outputs[3] * current_weight.unsqueeze(0))
                        final_predictions += mi_matrix[2, 3] * pair_pred

                    # 计算损失并更新权重
                    loss = criterion(final_predictions, batch_labels)
                    loss.backward()
                    optimizer.step()
                    # 计算准确率
                    _, predicted = torch.max(final_predictions, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                    total_loss += loss.item()

                epoch_acc = 100 * correct / total
                epoch_loss = total_loss / len(dataloader)
                print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
                print(f'Current weight vector: {current_weight.data}')
                logging.info(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
                logging.info(f'Current weight vector: {current_weight.data}')

            # 在训练完成后更新原始权重
            with torch.no_grad():
                weights[client_idx].data = current_weight.data.clone()
        print('test_shape', test_features[0].shape)
        print('test_label_shape', test_labels)
        test_CRNet(client_models, client_cluster_centers, test_features, test_labels, weights, device=self.device,
                   dataset=self.dataset, clip=clip)
        return client_models, client_cluster_centers

