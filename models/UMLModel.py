import copy
import gc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import functools

from torch.cuda.amp import GradScaler

from torch import autocast
from torch.utils.data import Dataset, DataLoader, random_split

from etc.utils import SES, SNS, HMS, TSPHead, remap_labels, map_tensor, NegCosineSimilarityLoss

from tqdm import tqdm

global TORCH_BACKEND_DEVICE  # torch 后端设备，可能是cpu、cuda、mps
if torch.cuda.is_available():
    TORCH_BACKEND_DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    TORCH_BACKEND_DEVICE = torch.device('mps')
else:
    TORCH_BACKEND_DEVICE = torch.device("cpu")


def contains_nan(tensor):
    return torch.isnan(tensor).any().item()

def log_mse_loss(pred, target):
    return torch.log(torch.mean((pred - target) ** 2) + 1e-8)




def nan_detector(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        x = args[0]
        for name, module in self.named_children():
            x = module(x)
            if contains_nan(x):
                print(f"NaN detected in the output of {name}")
                return x  # 返回 NaN 张量，中止前向传播
        return x

    return wrapper


class UMLFramework(nn.Module):
    def __init__(self, base_model, class_dict, num_tasks=100, mode='None',
                 N_way=3,
                 K_shot=5,
                 Q_query=7,
                 inner_step=2,
                 input_dim=14,
                 num_features=2,
                 ensemble_dim=512, output_dim=512,
                 data_type='tab',
                 device='cuda'):
        """
        通用元学习框架
        :param base_model: <torch.nn.Module> 基础模型
        :param class_dict: <dict> 分类映射字典
        :param num_tasks: <int> 每次元学习中的任务数量
        :param N_way: <int> 每个任务中的类别数量
        :param K_shot: <int> 每个类别的样本数量
        :param Q_query: <int> 每个类别的查询样本数量
        :param inner_step: <int> 内循环中的梯度更新步数
        :param ensemble_dim: <int> 输入维度
        :param output_dim: <int> 输出维度
        :param device: <torch.device> 训练设备
        """
        super(UMLFramework, self).__init__()
        self.device = TORCH_BACKEND_DEVICE
        self.base_model = base_model.to(self.device)
        self.input_dim = input_dim
        self.num_features = num_features
        self.ses = SES(num_tasks=num_tasks, N_way=N_way, K_shot=K_shot, Q_query=Q_query,
                       class_dict=class_dict)  # SES 实现
        self.sns = SNS().to(self.device)  # SNS 实现
        self.hms = HMS(beta=0.5).to(self.device)  # HMS 实现，beta是混合强度
        self.tsp_head = TSPHead(input_dim=ensemble_dim, output_dim=output_dim, num_heads=8).to(self.device)  # TSP-Head 实现
        self.optimizer =torch.optim.AdamW(self.base_model.parameters(),
                                            lr=1e-4,  # 使用较小的学习率
                                            weight_decay=0.01,
                                            eps=1e-8  # 增加数值稳定性
                                        )# optim.Adam(self.base_model.parameters(), lr=0.0005)
        # self.inner_optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.inner_steps = inner_step  # 内循环中的梯度更新步数
        self.scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=0.5,
                                                                    patience=5,
                                                                    verbose=True
                                                                )#  optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)  # 余弦退火调整学习
        self.loss_fn = nn.HuberLoss().to(self.device)  # log_mse_loss
        self.mse_loss_fn = nn.MSELoss().to(self.device)
        self.ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(self.device)# nn.CrossEntropyLoss().to(self.device)
        self.data_type = data_type
        nn.MSELoss().to(device) # nn.CrossEntropyLoss().to(device)
        # 增强几次
        self.K = 3

    def train_model(self, train_loader, test_loader=None, generated_loader=None, epoch=0):
        """
        元学习训练过程，使用混合精度训练和优化技巧
        """
        torch.cuda.empty_cache()
        gc.collect()
        self.train()
        scaler = GradScaler()  # 混合精度训练的scaler

        # 获取所有标签并创建映射
        all_labels = torch.cat([labels for _, labels in train_loader])
        all_labels = all_labels.to(self.device)
        unique_labels = all_labels.unique()
        label_to_index = {label.item(): idx for idx, label in enumerate(unique_labels)}
        padding_length = 1

        total_meta_loss = 0

        # 表格数据的处理
        if self.data_type == 'tab':
            for batch_idx, (features, labels) in enumerate(train_loader):
                meta_loss = 0
                features = features.to(self.device)
                labels = labels.to(self.device)

                indices = list(range(features.size(1)))
                random.shuffle(indices)

                aug_query_loss_total = 0
                aug_support_loss_total = 0

                # 特征增强循环
                for idx in range(self.K):
                    selected_col = indices[idx]
                    # 特征处理
                    selected_X = torch.cat((features[:, :selected_col],
                                            features[:, selected_col + 1:]), dim=1).to(self.device)
                    selected_X = F.pad(selected_X, (0, padding_length), "constant", 0)
                    selected_y = features[:, selected_col].view(-1, 1).to(self.device)

                    # 确定任务类型
                    unique_values = selected_y.unique()
                    task_type = self._determine_task_type(len(unique_values))

                    # 任务采样和训练
                    tasks = self.ses.sample_tasks_old(selected_X, selected_y)
                    for task in tasks:
                        with autocast(device_type=str(self.device), dtype=torch.float32):
                            meta_results = self._train_single_task(task, task_type, batch_idx, idx, scaler)

                        if meta_results is None:  # 如果发生错误
                            continue

                        aug_support_loss_total += meta_results['support_loss']
                        aug_query_loss_total += meta_results['query_loss']
                        meta_loss += meta_results['query_loss']

                total_meta_loss += meta_loss
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Meta Loss: {meta_loss:.4f}')

        # 使用生成数据或原始数据进行微调
        loader = generated_loader if generated_loader is not None else train_loader
        finetune_loss = self._finetune_phase(loader, scaler)

        # 评估阶段
        eval_metrics = self.evaluate_model(test_loader if test_loader is not None else train_loader)

        # 打印训练结果
        print(f'\nEpoch {epoch} Summary:')
        print(f'Meta Learning Loss: {total_meta_loss / len(train_loader):.4f}')
        print(f'Finetune Loss: {finetune_loss:.4f}')
        print(f'Test Accuracy: {eval_metrics["accuracy"]:.2f}%')

        return eval_metrics

    def evaluate_model(self, loader):
        """
        评估模型性能
        """
        self.eval()
        total = 0
        correct = 0
        total_loss = 0

        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(loader):
                if isinstance(features, list):
                    features = [f.to(self.device, dtype=torch.float32) for f in features]
                else:
                    features = features.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device)
                new_labels = torch.tensor([label.item() for label in labels]).to(self.device)

                with autocast(device_type=str(self.device), dtype=torch.float32):
                    preds = self.base_model(features, mode="client")
                    loss = self.ce_loss_fn(preds, new_labels)

                    _, predicted = torch.max(F.softmax(preds, dim=-1), 1)
                    total += labels.size(0)
                    correct += (predicted == new_labels).sum().item()
                    total_loss += loss.item()

        accuracy = (correct / total) * 100
        avg_loss = total_loss / len(loader)

        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }

    def _finetune_phase(self, loader, scaler):
        """
        微调阶段
        """
        total_loss = 0
        for batch_idx, (features, labels) in enumerate(loader):
            new_labels = torch.tensor([label.item() for label in labels])
            if self.data_type == 'tab':
                tasks = self.ses.sample_tasks_old(features, new_labels)
            else:
                tasks = self.ses.sample_tasks(features, new_labels)

            for task in tasks:
                support_x, support_y = self._prepare_task_data(task['support_images'], task['support_labels'])
                query_x, query_y = self._prepare_task_data(task['query_images'], task['query_labels'])

                # 克隆模型
                # cloned_model = copy.deepcopy(self.base_model)
                try:
                    # print(self.input_dim)
                    # print(self.num_features)
                    cloned_model = type(self.base_model)(feature_dim=self.input_dim, num_features=self.num_features).to(self.device)
                except:
                    cloned_model = type(self.base_model)(self.input_dim).to(self.device)  # 直接创建同类型的新模型
                cloned_model.load_state_dict(self.base_model.state_dict())
                cloned_model = cloned_model.to(self.device)
                cloned_model.train()
                inner_optimizer = optim.Adam(cloned_model.parameters(), lr=0.005)

                # 内循环训练
                for _ in range(self.inner_steps):
                    with autocast(device_type=str(self.device), dtype=torch.float32):
                        support_preds = cloned_model(support_x, mode='client')
                        support_loss = self.ce_loss_fn(support_preds, support_y.squeeze(-1).long())

                    cloned_model.zero_grad()
                    scaler.scale(support_loss).backward()
                    scaler.unscale_(inner_optimizer)
                    torch.nn.utils.clip_grad_norm_(cloned_model.parameters(), max_norm=1.0)
                    scaler.step(inner_optimizer)
                    scaler.update()

                # 外循环优化
                with autocast(device_type=str(self.device), dtype=torch.float32):
                    query_preds = cloned_model(query_x, mode='client')
                    query_loss = self.ce_loss_fn(query_preds, query_y.squeeze(-1).long())
                    total_loss += query_loss.item()

                self.optimizer.zero_grad()
                scaler.scale(query_loss).backward()

                # 更新基础模型参数
                for param, cloned_param in zip(self.base_model.parameters(), cloned_model.parameters()):
                    if param.grad is None:
                        param.grad = cloned_param.grad
                    elif cloned_param.grad is not None:
                        param.grad += cloned_param.grad

                del cloned_model
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()

        return total_loss / len(loader)

    def _train_single_task(self, task, task_type, batch_idx, idx, scaler):
        """单个任务的训练"""
        support_x, support_y = self._prepare_task_data(task['support_images'], task['support_labels'])
        query_x, query_y = self._prepare_task_data(task['query_images'], task['query_labels'])

        # 克隆模型
        # cloned_model = copy.deepcopy(self.base_model)
        try:
            cloned_model = type(self.base_model)(feature_dim=self.input_dim, num_features=self.num_features).to(self.device)
        except:
            cloned_model = type(self.base_model)(self.input_dim).to(self.device)  # 直接创建同类型的新模型
        cloned_model.load_state_dict(self.base_model.state_dict())
        cloned_model = cloned_model.to(self.device)
        cloned_model.train()
        inner_optimizer = optim.Adam(cloned_model.parameters(), lr=0.005)

        # 内循环优化
        for inner_step in range(self.inner_steps):
            with autocast(device_type=str(self.device), dtype=torch.float32):
                support_preds = cloned_model(support_x, mode=task_type)
                if contains_nan(support_preds):
                    print(f"NaN detected in support predictions at batch {batch_idx}, inner step {inner_step}")
                    return None

                support_loss = self._compute_loss(support_preds, support_y, task_type)

            # 使用scaler进行反向传播
            cloned_model.zero_grad()
            scaler.scale(support_loss).backward()
            scaler.unscale_(inner_optimizer)
            torch.nn.utils.clip_grad_norm_(cloned_model.parameters(), max_norm=1.0)
            scaler.step(inner_optimizer)
            scaler.update()

        # 外循环优化
        with autocast(device_type=str(self.device), dtype=torch.float32):
            query_preds = cloned_model(query_x, mode=task_type)
            query_loss = self._compute_loss(query_preds, query_y, task_type)

        self.optimizer.zero_grad()
        scaler.scale(query_loss).backward()

        # 更新基础模型参数
        for param, cloned_param in zip(self.base_model.parameters(), cloned_model.parameters()):
            if param.grad is None:
                param.grad = cloned_param.grad
            elif cloned_param.grad is not None:
                param.grad += cloned_param.grad

        del cloned_model
        scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
        scaler.step(self.optimizer)
        scaler.update()

        return {
            'support_loss': support_loss.item(),
            'query_loss': query_loss.item()
        }

    def _prepare_task_data(self, x, y):
        """准备任务数据"""
        if isinstance(x, list):
            x = [x_.to(self.device, dtype=torch.float32) for x_ in x]
        else:
            x = x.to(self.device, dtype=torch.float32)
        y = y.to(self.device)
        return x, y

    def _compute_loss(self, preds, targets, task_type):
        """计算损失"""
        if 'Cls' in task_type:
            return self.ce_loss_fn(preds, targets.squeeze(-1).long())
        elif 'Regression' in task_type:
            return self.mse_loss_fn(preds.squeeze(-1), targets.squeeze(-1))
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _determine_task_type(self, num_unique_values):
        """确定任务类型"""
        if num_unique_values <= 2:
            return 'BinaryCls'
        elif num_unique_values <= 6:
            return f'{num_unique_values}Cls'
        else:
            return 'Regression'

    def train_model_ol(self, train_loader, test_loader=None, generated_loader=None, epoch=0):
        """
        这一训练函数用于特征提取器ISLCmodel的元学习训练过程
        参考MAML框架
        :param train_loader:
        :param epoch:
        :return:
        """
        self.train()
        all_labels = torch.cat([labels for _, labels in train_loader])
        all_labels = all_labels.to(self.device)
        unique_labels = all_labels.unique()
        # # 为每个唯一标签分配一个从0开始的全局索引
        label_to_index = {label.item(): idx for idx, label in enumerate(unique_labels)}
        padding_length = 1  # 由于删除了一列特征，需要在末尾添加一列\
        if self.data_type == 'tab':
            for batch_idx, (features, labels) in enumerate(train_loader):
                meta_loss = 0
                meta_accuracy = 0
                features, labels = features.to(self.device), labels.to(self.device)
                # print("labels shape", labels.shape)
                indices = list(range(features.size(1)))
                random.shuffle(indices)
                # 类别增强，所有样本的其他列拿来当标签，用增强的数据做元训练阶段（哪个是元训练阶段），用真实类别的样本做元测试阶段（哪个是元测试阶段）
                # 类别增强的数据也分support set和query set, 也分内循环外循环
                # 类别增强的数据训练完以后，再用真实数据训练，也分support set和query set, 不分内外循环，正常训练、测试
                # support_set训练用来初始化，元训练阶段，增强数据，query_set
                aug_query_loss_total = 0
                aug_support_loss_total = 0
                for idx in range(self.K):
                    selected_col = indices[idx]
                    # 将特征和标签分离，selected_col 为标签，其余列为特征
                    selected_X = torch.cat((features[:, :selected_col],
                                            features[:, selected_col + 1:]), dim=1).to(self.device)
                    # 计算需要padding的长度
                    # 对特征进行padding,保证数据增强前后不改变样本特征
                    selected_X = F.pad(selected_X, (0, padding_length), "constant", 0)
                    selected_y = features[:, selected_col].view(-1, 1).to(self.device)
                    # print('sel', selected_y)
                    unique_values = selected_y.unique()
                    num_unique_values = len(unique_values)
                    if num_unique_values == 1 or num_unique_values == 2:
                        task_type = 'BinaryCls'
                    elif num_unique_values == 3:
                        task_type = 'ThreeCls'
                    elif num_unique_values == 4:
                        task_type = 'FourCls'
                    elif num_unique_values == 5:
                        task_type = 'FiveCls'
                    elif num_unique_values == 6:
                        task_type = 'SixCls'
                    elif num_unique_values >= 7:
                        task_type = 'Regression'
                    else:
                        print('wrong')
                        print(unique_values)
                        print(num_unique_values)
                        raise ValueError

                    tasks = self.ses.sample_tasks_old(selected_X, selected_y)
                    for task in tasks:
                        support_x = task['support_images']
                        # print('support x shape', support_x.shape)
                        support_y = task['support_labels']
                        # support_ind = task['support_label_indices']
                        query_x = task['query_images']
                        # print('query x shape', query_x.shape)
                        query_y = task['query_labels']
                        # query_ind = task['query_label_indices']
                        # 将支持集数据移动到设备上
                        support_x, support_y = support_x.to(self.device), support_y.to(self.device)
                        query_x, query_y = query_x.to(self.device), query_y.to(self.device)
                        # support_ind = support_ind.to(self.device)
                        # query_ind = query_ind.to(self.device)
                        # 克隆初始模型参数
                        cloned_model = copy.deepcopy(self.base_model)
                        cloned_model.train()
                        inner_optimizer = optim.Adam(cloned_model.parameters(), lr=0.005)
                        for _ in range(self.inner_steps):
                            support_preds = cloned_model(support_x, mode=task_type)  # , support_ind)
                            if contains_nan(support_preds):
                                print(f"NaN detected in model output at batch {batch_idx}, Syn{idx}")
                                break
                            # print(support_preds.shape, support_y.shape)
                            if 'Cls' in task_type:
                                support_loss = self.ce_loss_fn(support_preds, support_y.squeeze(-1).long())
                            elif 'Regression' in task_type:
                                support_loss = self.mse_loss_fn(support_preds.squeeze(-1), support_y.squeeze(-1))
                            else:
                                raise ValueError
                            aug_support_loss_total += support_loss
                            cloned_model.zero_grad()
                            support_loss.backward()
                            torch.nn.utils.clip_grad_norm_(cloned_model.parameters(), max_norm=1.0)
                            inner_optimizer.step()

                        query_preds = cloned_model(query_x, mode=task_type)
                        if contains_nan(query_preds):
                            print(f"NaN detected in query predictions at batch {batch_idx}, syn{idx}")
                            return
                        if contains_nan(query_y):
                            print(f"NaN detected in query labels at batch {batch_idx}, syn{idx}")
                            return
                        if len(query_y) == 0:
                            continue
                        if 'Cls' in task_type:
                            query_loss = self.ce_loss_fn(query_preds, query_y.squeeze(-1).long())
                        elif 'Regression' in task_type:
                            query_loss = self.mse_loss_fn(query_preds.squeeze(-1), query_y.squeeze(-1))
                        else:
                            raise ValueError
                        aug_query_loss_total += query_loss
                        if torch.isnan(query_loss):
                            print(f"NaN detected in query loss at batch {batch_idx}, syn{idx}")
                            print('query_loss', query_preds)
                            print('query_y', query_y)
                            return

                        # 外循环：通过查询集的损失反向传播更新初始模型参数
                        self.optimizer.zero_grad()
                        query_loss.backward()

                        # 确保 basemodel 的参数得到更新
                        for param, cloned_param in zip(self.base_model.parameters(), cloned_model.parameters()):
                            if param.grad is None:
                                param.grad = cloned_param.grad
                            elif cloned_param.grad is None:
                                param.grad = param.grad
                            elif contains_nan(param.grad):
                                print(f"NaN detected in meta gradient of {param} at batch {batch_idx}")
                                return
                            else:
                                # print(type(param.grad), type(cloned_param.grad))
                                param.grad += cloned_param.grad
                        torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
                        if contains_nan(query_preds):
                            print(f"NaN detected in query predictions at batch {batch_idx}")
                            return

                        self.optimizer.step()

        # 如果使用生成数据，则只使用具有标签的生成数据进行微调
        if generated_loader is None:
            loader = train_loader
        else:
            loader = generated_loader

        total_loss = 0
        for batch_idx, (features, labels) in enumerate(loader):
            # print(1)
            # print(features, labels)
            new_labels = torch.tensor([label.item() for label in labels])
            if self.data_type == 'tab':
                tasks = self.ses.sample_tasks_old(features, new_labels)
            else:
                tasks = self.ses.sample_tasks(features, new_labels)
            # tasks = self.ses.sample_tasks(features, new_labels)
            # print("num tasks", len(tasks))
            # TODO：类别数量不定的task
            # query和support类别都不一定且不重合
            for task in tasks:
                # print(2)
                support_x = task['support_images']
                support_y = task['support_labels']
                # print('sup0', support_x[0].shape)
                # support_ind = task['support_label_indices']
                query_x = task['query_images']
                query_y = task['query_labels']
                # query_ind = task['query_label_indices']
                # 将支持集数据移动到设备上
                support_x, support_y = [support_x1.to(self.device).half()  for support_x1 in support_x], support_y.to(self.device)
                query_x, query_y = [query_x1.to(self.device).half()  for query_x1 in query_x], query_y.to(self.device)
                # support_ind = support_ind.to(self.device)
                # query_ind = query_ind.to(self.device)
                # 克隆初始模型参数
                cloned_model = copy.deepcopy(self.base_model)
                cloned_model.train()
                inner_optimizer = optim.Adam(cloned_model.parameters(), lr=0.005)
                for inner_step in range(self.inner_steps):
                    support_preds = cloned_model(support_x, mode='client')  # , support_ind)
                    if contains_nan(support_preds):
                        print(f"NaN detected in support predictions at batch {batch_idx}, inner step {inner_step}")
                    # support_y = support_y.unsqueeze(1)
                    # print(support_preds.shape, support_y.shape)
                    support_loss = self.ce_loss_fn(support_preds, support_y.squeeze(-1).long())
                    cloned_model.zero_grad()
                    support_loss.backward()
                    inner_optimizer.step()

                query_preds = cloned_model(query_x, mode='client')
                query_loss = self.ce_loss_fn(query_preds, query_y.squeeze(-1).long())
                total_loss += query_loss
                # 外循环：通过查询集的损失反向传播更新初始模型参数
                self.optimizer.zero_grad()
                query_loss.backward()
                # 确保 basemodel 的参数得到更新
                for param, cloned_param in zip(self.base_model.parameters(), cloned_model.parameters()):
                    if param.grad is None:
                        param.grad = cloned_param.grad
                    elif cloned_param.grad is None:
                        param.grad = param.grad
                    else:
                        param.grad += cloned_param.grad
                torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
                self.optimizer.step()

        total_correct = 0
        total_y = 0
        if test_loader is not None:
            loader = test_loader
        self.eval()
        for batch_idx, (features, labels) in enumerate(loader):
            new_labels = torch.tensor([label.item() for label in labels])
            if self.data_type == 'tab':
                tasks = self.ses.sample_tasks_old(features, new_labels)
            else:
                tasks = self.ses.sample_tasks(features, new_labels)
            # tasks = self.ses.sample_tasks(features, new_labels)
            for task in tasks:
                query_x = task['query_images']
                query_y = task['query_labels']
                query_x, query_y = [query_x1.to(self.device).half()  for query_x1 in query_x], query_y.to(self.device)
                query_preds = self.base_model(query_x, mode='client')  # query_ind)
                _, predicted = torch.max(F.softmax(query_preds, dim=-1), 1)
                correct = (predicted == query_y).sum().item()
                total = query_y.size(0)
                total_correct += correct
                total_y += total

        if total_y > 0:
            print(
                f'Epoch: {epoch}, After Class augmentation, Total Meta Loss: {total_loss}, Meta Task Test Accuracy: {(total_correct / total_y) * 100}%')

    def train_model_old(self, train_loader, test_loader=None, generated_loader=None, epoch=0):
        """
        这一训练函数用于特征提取器ISLCmodel的元学习训练过程
        参考MAML框架
        :param train_loader:
        :param epoch:
        :return:
        """
        self.train()
        all_labels = torch.cat([labels for _, labels in train_loader])
        all_labels = all_labels.to(self.device)
        unique_labels = all_labels.unique()
        # # 为每个唯一标签分配一个从0开始的全局索引
        label_to_index = {label.item(): idx for idx, label in enumerate(unique_labels)}
        padding_length = 1  # 由于删除了一列特征，需要在末尾添加一列\
        for batch_idx, (features, labels) in enumerate(train_loader):
            meta_loss = 0
            meta_accuracy = 0
            features, labels = features.to(self.device), labels.to(self.device)
            # print("labels shape", labels.shape)
            indices = list(range(features.size(1)))
            random.shuffle(indices)
            # 类别增强，所有样本的其他列拿来当标签，用增强的数据做元训练阶段（哪个是元训练阶段），用真实类别的样本做元测试阶段（哪个是元测试阶段）
            # 类别增强的数据也分support set和query set, 也分内循环外循环
            # 类别增强的数据训练完以后，再用真实数据训练，也分support set和query set, 不分内外循环，正常训练、测试
            # support_set训练用来初始化，元训练阶段，增强数据，query_set
            aug_query_loss_total = 0
            aug_support_loss_total = 0
            for idx in range(self.K):
                selected_col = indices[idx]
                # 将特征和标签分离，selected_col 为标签，其余列为特征
                selected_X = torch.cat((features[:, :selected_col],
                                        features[:, selected_col + 1:]), dim=1).to(self.device)
                # 计算需要padding的长度
                # 对特征进行padding,保证数据增强前后不改变样本特征
                selected_X = F.pad(selected_X, (0, padding_length), "constant", 0)
                selected_y = features[:, selected_col].view(-1, 1).to(self.device)
                unique_values = selected_y.unique()
                num_unique_values = len(unique_values)
                if num_unique_values == 1 or num_unique_values == 2:
                    task_type = 'BinaryCls'
                elif num_unique_values == 3:
                    task_type = 'ThreeCls'
                elif num_unique_values == 4:
                    task_type = 'FourCls'
                elif num_unique_values == 5:
                    task_type = 'FiveCls'
                elif num_unique_values == 6:
                    task_type = 'SixCls'
                elif num_unique_values >= 7:
                    task_type = 'Regression'
                else:
                    print('wrong')
                    print(unique_values)
                    print(num_unique_values)
                    raise ValueError

                tasks = self.ses.sample_tasks_old(selected_X, selected_y)
                for task in tasks:
                    support_x = task['support_images']
                    # print('support x shape', support_x.shape)
                    support_y = task['support_labels']
                    # support_ind = task['support_label_indices']
                    query_x = task['query_images']
                    # print('query x shape', query_x.shape)
                    query_y = task['query_labels']
                    # query_ind = task['query_label_indices']
                    # 将支持集数据移动到设备上
                    support_x, support_y = support_x.to(self.device), support_y.to(self.device)
                    query_x, query_y = query_x.to(self.device), query_y.to(self.device)
                    # support_ind = support_ind.to(self.device)
                    # query_ind = query_ind.to(self.device)
                    # 克隆初始模型参数
                    cloned_model = copy.deepcopy(self.base_model)
                    cloned_model.train()
                    inner_optimizer = optim.Adam(cloned_model.parameters(), lr=0.005)
                    for _ in range(self.inner_steps):
                        support_preds = cloned_model(support_x, mode=task_type)  # , support_ind)
                        if contains_nan(support_preds):
                            print(f"NaN detected in model output at batch {batch_idx}, Syn{idx}")
                            break
                        # print(support_preds.shape, support_y.shape)
                        if 'Cls' in task_type:
                            support_loss = self.ce_loss_fn(support_preds, support_y.squeeze(-1).long())
                        elif 'Regression' in task_type:
                            support_loss = self.mse_loss_fn(support_preds.squeeze(-1), support_y.squeeze(-1))
                        else:
                            raise ValueError
                        aug_support_loss_total += support_loss
                        cloned_model.zero_grad()
                        support_loss.backward()
                        torch.nn.utils.clip_grad_norm_(cloned_model.parameters(), max_norm=1.0)
                        inner_optimizer.step()

                    query_preds = cloned_model(query_x, mode=task_type)
                    if contains_nan(query_preds):
                        print(f"NaN detected in query predictions at batch {batch_idx}, syn{idx}")
                        return
                    if contains_nan(query_y):
                        print(f"NaN detected in query labels at batch {batch_idx}, syn{idx}")
                        return
                    if len(query_y) == 0:
                        continue
                    if 'Cls' in task_type:
                        query_loss = self.ce_loss_fn(query_preds, query_y.squeeze(-1).long())
                    elif 'Regression' in task_type:
                        query_loss = self.mse_loss_fn(query_preds.squeeze(-1), query_y.squeeze(-1))
                    else:
                        raise ValueError
                    aug_query_loss_total += query_loss
                    if torch.isnan(query_loss):
                        print(f"NaN detected in query loss at batch {batch_idx}, syn{idx}")
                        print('query_loss', query_preds)
                        print('query_y', query_y)
                        return

                    # 外循环：通过查询集的损失反向传播更新初始模型参数
                    self.optimizer.zero_grad()
                    query_loss.backward()

                    # 确保 basemodel 的参数得到更新
                    for param, cloned_param in zip(self.base_model.parameters(), cloned_model.parameters()):
                        if param.grad is None:
                            param.grad = cloned_param.grad
                        elif cloned_param.grad is None:
                            param.grad = param.grad
                        elif contains_nan(param.grad):
                            print(f"NaN detected in meta gradient of {param} at batch {batch_idx}")
                            return
                        else:
                            # print(type(param.grad), type(cloned_param.grad))
                            param.grad += cloned_param.grad
                    torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
                    if contains_nan(query_preds):
                        print(f"NaN detected in query predictions at batch {batch_idx}")
                        return

                    self.optimizer.step()

        # 如果使用生成数据，则只使用具有标签的生成数据进行微调
        if generated_loader is None:
            loader = train_loader
        else:
            loader = generated_loader

        total_loss = 0
        for batch_idx, (features, labels) in enumerate(loader):
            # print(1)
            # print(features, labels)
            new_labels = torch.tensor([label.item() for label in labels])
            tasks = self.ses.sample_tasks_old(features, new_labels)
            # print("num tasks", len(tasks))
            # TODO：类别数量不定的task
            # query和support类别都不一定且不重合
            for task in tasks:
                # print(2)
                support_x = task['support_images']
                support_y = task['support_labels']
                # support_ind = task['support_label_indices']
                query_x = task['query_images']
                query_y = task['query_labels']
                # query_ind = task['query_label_indices']
                # 将支持集数据移动到设备上
                support_x, support_y = support_x.to(self.device), support_y.to(self.device)
                query_x, query_y = query_x.to(self.device), query_y.to(self.device)
                # support_ind = support_ind.to(self.device)
                # query_ind = query_ind.to(self.device)
                # 克隆初始模型参数
                cloned_model = copy.deepcopy(self.base_model)
                cloned_model.train()
                inner_optimizer = optim.Adam(cloned_model.parameters(), lr=0.005)
                for inner_step in range(self.inner_steps):
                    support_preds = cloned_model(support_x, mode='client')  # , support_ind)
                    if contains_nan(support_preds):
                        print(f"NaN detected in support predictions at batch {batch_idx}, inner step {inner_step}")
                    # support_y = support_y.unsqueeze(1)
                    # print(support_preds.shape, support_y.shape)
                    support_loss = self.ce_loss_fn(support_preds, support_y.squeeze(-1).long())
                    cloned_model.zero_grad()
                    support_loss.backward()
                    inner_optimizer.step()

                query_preds = cloned_model(query_x, mode='client')
                query_loss = self.ce_loss_fn(query_preds, query_y.squeeze(-1).long())
                total_loss += query_loss
                # 外循环：通过查询集的损失反向传播更新初始模型参数
                self.optimizer.zero_grad()
                query_loss.backward()
                # 确保 basemodel 的参数得到更新
                for param, cloned_param in zip(self.base_model.parameters(), cloned_model.parameters()):
                    if param.grad is None:
                        param.grad = cloned_param.grad
                    elif cloned_param.grad is None:
                        param.grad = param.grad
                    else:
                        param.grad += cloned_param.grad
                torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
                self.optimizer.step()

        total_correct = 0
        total_y = 0
        if test_loader is not None:
            loader = test_loader
        self.eval()
        for batch_idx, (features, labels) in enumerate(loader):
            new_labels = torch.tensor([label.item() for label in labels])
            tasks = self.ses.sample_tasks_old(features, new_labels)
            for task in tasks:
                query_x = task['query_images']
                query_y = task['query_labels']
                query_x, query_y = query_x.to(self.device), query_y.to(self.device)
                query_preds = self.base_model(query_x, mode='client')  # query_ind)
                _, predicted = torch.max(F.softmax(query_preds, dim=-1), 1)
                correct = (predicted == query_y).sum().item()
                total = query_y.size(0)
                total_correct += correct
                total_y += total

        if total_y > 0:
            print(
                f'Epoch: {epoch}, After Class augmentation, Total Meta Loss: {total_loss}, Meta Task Test Accuracy: {(total_correct / total_y) * 100}%')

    def fine_tune_without_test(self, train_loader, test_loader=None, generated_loader=None, epoch=0):
        self.base_model.train()

        # 使用混合精度训练
        scaler = GradScaler()

        for batch_idx, (features, labels) in enumerate(train_loader):
            features = [feature.to(self.device, dtype=torch.float32) for feature in features]
            labels = labels.to(self.device)
            new_labels = torch.tensor([label.item() for label in labels]).to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type=str(self.device), dtype=torch.float32):
                preds = self.base_model(features, mode="client")
                loss = self.ce_loss_fn(preds, new_labels)

                if torch.isnan(loss):
                    print(f"NaN loss detected at batch {batch_idx}")
                    print(f"Predictions stats: min={preds.min()}, max={preds.max()}, mean={preds.mean()}")
                    continue

            scaler.scale(loss).backward()

            # 梯度裁剪
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)

            scaler.step(self.optimizer)
            scaler.update()

    def fine_tune(self, train_loader, test_loader=None, generated_loader=None, epoch=0):
        self.base_model.train()
        scaler = GradScaler()

        # 用于记录每个epoch的总损失
        running_loss = 0.0
        total_batches = 0

        # 训练阶段
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = [feature.to(self.device, dtype=torch.float32) for feature in features]
            labels = labels.to(self.device)
            new_labels = torch.tensor([label.item() for label in labels]).to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type=str(self.device), dtype=torch.float32):
                preds = self.base_model(features, mode="client")
                loss = self.ce_loss_fn(preds, new_labels)

                if torch.isnan(loss):
                    print(f"NaN loss detected at batch {batch_idx}")
                    print(f"Predictions stats: min={preds.min()}, max={preds.max()}, mean={preds.mean()}")
                    continue

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
            scaler.step(self.optimizer)
            scaler.update()

            # 累计损失
            running_loss += loss.item()
            total_batches += 1

            # 每N个batch打印训练信息
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Training Loss: {loss.item():.4f}')

        # 计算平均训练损失
        avg_train_loss = running_loss / total_batches if total_batches > 0 else 0

        # 评估阶段
        self.base_model.eval()
        total_test = 0
        correct_test = 0
        test_loss = 0.0

        # 决定使用哪个数据加载器进行评估
        eval_loader = test_loader if test_loader is not None else train_loader

        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(eval_loader):
                features = [feature.to(self.device, dtype=torch.float32) for feature in features]
                labels = labels.to(self.device)
                new_labels = torch.tensor([label.item() for label in labels]).to(self.device)

                with autocast(device_type=str(self.device), dtype=torch.float32):
                    preds = self.base_model(features, mode="client")
                    loss = self.ce_loss_fn(preds, new_labels)

                    # 计算准确率
                    _, predicted = torch.max(F.softmax(preds, dim=-1), 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == new_labels).sum().item()
                    test_loss += loss.item()

        # 计算评估指标
        accuracy = (correct_test / total_test) * 100
        avg_test_loss = test_loss / len(eval_loader)

        # 打印详细的训练和评估结果
        print(f'\nEpoch {epoch} Summary:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Test Loss: {avg_test_loss:.4f}')
        print(f'Test Accuracy: {accuracy:.2f}%')
        print(f'Correct/Total: {correct_test}/{total_test}')

        # 如果有学习率调度器，可以在这里更新
        if hasattr(self, 'scheduler'):
            self.scheduler.step(avg_test_loss)

        return {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'accuracy': accuracy,
            'correct': correct_test,
            'total': total_test
        }

    def fine_tune_old(self, train_loader, test_loader=None, generated_loader=None, epoch=0):
        """
        这一训练函数用于特征提取器ISLCmodel的微调过程
        参考MAML框架
        :param train_loader:
        :param epoch:
        :return:
        """
        self.base_model.train()
        all_labels = torch.cat([labels for _, labels in train_loader])
        all_labels = all_labels.to(self.device)
        unique_labels = all_labels.unique()
        # 为每个唯一标签分配一个从0开始的全局索引
        label_to_index = {label.item(): idx for idx, label in enumerate(unique_labels)}
        total_loss = 0
        for batch_idx, (features, labels) in enumerate(train_loader):
            features =features.to(self.device) #for feature in features]
            # print('fea0', features[0].shape)
            labels = labels.to(self.device)
            # 使用全局映射将标签转换为从0开始索引的新标签
            new_labels = torch.tensor([label.item() for label in labels]).to(self.device)
            preds = self.base_model(features, mode="client")
            loss = self.ce_loss_fn(preds, new_labels)
            total_loss += loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.base_model.eval()
        total = 0
        correct = 0
        if test_loader is not None:
            loader = test_loader
        else:
            loader = train_loader
        for batch_idx, (features, labels) in enumerate(loader):
            features = features.to(self.device)
            # print('fea0', features[0].shape)
            labels = labels.to(self.device)
            new_labels = torch.tensor([label.item() for label in labels]).to(self.device)
            preds = self.base_model(features, mode="client")
            _, predicted = torch.max(F.softmax(preds, dim=-1), 1)
            correct += (predicted == new_labels).sum().item()
            total += labels.size(0)
            # print("labels:", new_labels)
            # print("predicted:", predicted)
        accuracy = correct / total
        print(f'Epoch {epoch}, finetune, Loss: {total_loss}, Finetune Task Test Accuracy: {accuracy * 100}%')

    def train_old_model(self, train_loader, epoch=0):
        self.train()
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            print('train one batch start')
            loss = 0
            images, labels = images.to(self.device), labels.to(self.device)
            if self.data_type == 'tab':
                tasks = self.ses.sample_tasks_old(images, labels)
            else:
                tasks = self.ses.sample_tasks(images, labels)
            # tasks = self.ses.sample_tasks(images, labels)
            # TODO：类别数量不定的task
            # query和support类别都不一定且不重合
            for task in tasks:
                support_x = task['support_images']
                support_y = task['support_labels']
                query_x = task['query_images']
                query_y = task['query_labels']
                embedding_s = self.base_model(support_x)
                embedding_q = self.base_model(query_x)
                # print(embedding_q.shape)
                # 将query集中的embedding和查询集混合
                # TODO：类别不重合
                # TODO： 语义扩充
                mixed_embedding_q = self.hms.mix_supports(embedding_q, embedding_s)
                # print('mxe shape', mixed_embedding.shape)
                output_q = self.tsp_head(mixed_embedding_q)
                output_s = self.tsp_head(embedding_s)

                # print(output_s.shape)
                # print(output_q.shape)
                # 原需要计算支持集中每个类别中心，但是默认支持集为1，即不用计算
                # # 计算支持集中每个类别的中心
                # class_centers = []
                # for class_id in torch.unique(support_y):
                #     class_embeddings = output_s[support_y == class_id]
                #     class_center = class_embeddings.mean(dim=0)
                #     class_centers.append(class_center)
                # class_centers = torch.stack(class_centers)
                class_centers = output_s
                # TODO:更改相似度函数
                similarities = self.sns(output_q, class_centers)
                # 使用softmax将相似度转换为预测概率
                predictions = F.softmax(similarities, dim=1)
                # print(predictions.shape)
                # print(query_y)
                # TODO:remap函数是否会影响模型？
                query_y = remap_labels(query_y)
                # 计算交叉熵损失
                loss += F.cross_entropy(predictions, query_y)
                # 评估准确率
                _, predicted = torch.max(predictions.data, 1)
                total += query_y.size(0)
                correct += (predicted == query_y).sum().item()

            loss /= len(tasks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        accuracy = 100 * correct / total
        print(f'Epoch{epoch}: Accuracy of the model on the train images: {accuracy}%')
        self.scheduler.step()
        print('Scheduler step')


    # def evaluate_model(self, test_loader, epoch=0, mode='test'):
    #     self.eval()  # 将模型设置为评估模式
    #     correct = 0
    #     total = 0
    #
    #     with torch.no_grad():  # 在评估阶段不计算梯度
    #         for images, labels in test_loader:
    #             images, labels = images.to(self.device), labels.to(self.device)
    #             outputs = self(images)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
    #
    #     accuracy = 100 * correct / total
    #     print(f'Epoch{epoch}: Accuracy of the model on the {mode} images: {accuracy}%')






