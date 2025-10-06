import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, precision_recall_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
warnings.filterwarnings('ignore')


eps = 1e-7
class RCELoss(nn.Module):
    def __init__(self, scale=1.0):
        super(RCELoss, self).__init__()
        self.scale = scale

    def forward(self, pred, labels, sample_weights=None):
        labels = labels.float()
        pred = torch.sigmoid(pred)
        pred = torch.clamp(pred, min=eps, max=1.0)
        loss = -torch.mean(labels * torch.log(pred) + (1 - labels) * torch.log(1 - pred))

        # 如果提供了样本权重，则将其应用到损失上
        if sample_weights is not None:
            loss = loss * sample_weights

        return self.scale * loss.mean()

# class WeightedBCEWithLogitsLoss(nn.Module):
#     def __init__(self, pos_weight=None, reduction='mean', delta = 5e-5):
#         """
#         :param pos_weight: A weight of positive examples.
#         :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
#         """
#         super(WeightedBCEWithLogitsLoss, self).__init__()
#         self.register_buffer('pos_weight', pos_weight)
#         self.reduction = reduction
#         self.delta = delta
#
#     def forward(self, input, target, model, sample_weights):
#         if not (target.size() == input.size()):
#             raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))
#
#         # Calculate binary cross entropy with logits
#         loss = F.binary_cross_entropy_with_logits(input, target, pos_weight=self.pos_weight, reduction='none')
#
#         # Apply the sample weights
#         if sample_weights is not None:
#             loss = loss * sample_weights
#         l1_norm = sum(p.abs().sum() for p in model.parameters())
#
#         # Apply reduction
#         if self.reduction == 'mean':
#             return loss.mean() + self.delta * l1_norm
#         elif self.reduction == 'sum':
#             return loss.sum() + self.delta * l1_norm
#         else:
#             return loss + self.delta * l1_norm


# class NormalizedWeightedBCEWithLogitsLoss(nn.Module):
#     def __init__(self, pos_weight=None, reduction='mean', delta=5e-5):
#         """
#         :param pos_weight: A weight of positive examples.
#         :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
#         :param delta: Regularization parameter for L1 norm.
#         """
#         super(NormalizedWeightedBCEWithLogitsLoss, self).__init__()
#         self.register_buffer('pos_weight', pos_weight)
#         self.reduction = reduction
#         self.delta = delta
#
#     def forward(self, input, target, model, sample_weights):
#         if not (target.size() == input.size()):
#             raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))
#
#         # Calculate binary cross entropy with logits
#         loss = F.binary_cross_entropy_with_logits(input, target, pos_weight=self.pos_weight, reduction='none')
#
#         # Apply the sample weights
#         if sample_weights is not None:
#             loss = loss * sample_weights
#
#         # Calculate the L1 norm of model parameters
#         l1_norm = sum(p.abs().sum() for p in model.parameters())
#
#         # Normalize the loss
#         norm_factor = input.size(0)  # Use batch size as normalization factor
#         loss = loss / norm_factor
#
#         # Apply reduction
#         if self.reduction == 'mean':
#             return loss.mean() + self.delta * l1_norm / norm_factor
#         elif self.reduction == 'sum':
#             return loss.sum() / norm_factor + self.delta * l1_norm / norm_factor
#         else:
#             return loss + self.delta * l1_norm / norm_factor


class NormalizedWeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, reduction='mean', delta=5e-5):
        """
        :param pos_weight: 正样本的权重
        :param reduction: 用于指定损失的归约方式：'none' | 'mean' | 'sum'
        :param delta: 用于 L1 正则化的参数
        """
        super(NormalizedWeightedBCEWithLogitsLoss, self).__init__()
        self.register_buffer('pos_weight', pos_weight)
        self.reduction = reduction
        self.delta = delta

    def forward(self, input, target, model, sample_weights):
        # 确保输入和目标的尺寸相同
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        # 计算带有 logits 的二元交叉熵损失
        loss = F.binary_cross_entropy_with_logits(input, target, pos_weight=self.pos_weight, reduction='none')
        # print('loss', loss)
        # print('weight', sample_weights)
        # 应用样本权重
        if sample_weights is not None:
            loss = loss * sample_weights

        # 计算模型参数的 L1 范数用于正则化
        l1_norm = sum(p.abs().sum() for p in model.parameters())

        # 计算归一化因子（normalizor），这里将所有样本的损失进行求和
        normalizor = loss.sum()  # 使用损失的总和作为归一化因子

        # 用归一化因子对损失进行归一化
        loss = loss / normalizor

        # 根据 reduction 参数选择不同的归约方式
        if self.reduction == 'mean':
            # 如果是 'mean' 方式，返回平均损失并加上 L1 正则化
            return loss.mean() + self.delta * l1_norm / normalizor
        elif self.reduction == 'sum':
            # 如果是 'sum' 方式，返回损失的总和并加上 L1 正则化
            return loss.sum() / normalizor + self.delta * l1_norm / normalizor
        else:
            # 如果是 'none'，返回未归约的损失并加上 L1 正则化
            return loss + self.delta * l1_norm / normalizor


class ActiveNegativeLoss(torch.nn.Module):
    def __init__(self, active_loss, negative_loss,
                 alpha=5., beta=5., delta=5e-5) -> None:
        super().__init__()
        self.active_loss = active_loss
        self.negative_loss = negative_loss
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def forward(self, pred, labels, model, weights):
        al = self.active_loss(pred, labels, model, weights)
        al = al / pred.numel()  # 除以输入张量的元素数量
        nl = self.negative_loss(pred, labels)
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        # print('al', al)
        # print('nl', nl)
        # print(self.alpha, self.beta)
        loss = self.alpha * al + self.beta * nl + self.delta * l1_norm

        return loss


# 定义残差网络，两个残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.attention = Attention(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_channels)
                                          )

    def forward(self, x):
        indetity = x
        # print(indetity.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # print(out.shape)

        # out = F.pad(out, (0, 0, 0, 1))
        if indetity.shape != out.shape:
            indetity = self.shortcut(indetity)
        out += indetity # 残差连接
        out = self.relu(out)
        # out = self.attention(out)

        return out


class ConvModule(nn.Module):
    def __init__(self):
        super(ConvModule, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32,
                                            kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(inplace=True),
                                  )

        self.res_block1 = ResidualBlock(32, 32, 1)
        self.res_block2 = ResidualBlock(32, 32, 1)

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32,
                                             kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True)
                                   )

        self.res_block3 = ResidualBlock(32, 32, 1)
        self.res_block4 = ResidualBlock(32, 32, 1)

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64,
                                             kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))

        self.fc = nn.Sequential(nn.Dropout(p=0.2),
                                 nn.Linear(64 * sizes * sizes, 1),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        x = x.view(-1, 1, sizes, sizes)
        out = self.conv1(x)
        out = self.res_block1(out)
        out = self.res_block2(out)

        out = self.conv2(out)
        out = self.res_block3(out)
        out = self.res_block4(out)

        out = self.conv3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def train_epoch(net, data_loader, loss_func, optimizer):
    net.train() # 训练模式
    train_batch_num = len(data_loader)
    total_loss = 0 # 记录loss
    correct = 0 #正确分类样本数
    sample_sum = 0 # 样本总数
    train_pred, train_target = [], []

    for batch_idx, (data, target, sample_weights) in enumerate(data_loader):
        data = data.float()
        target = target.long()
        optimizer.zero_grad()
        output = net(data) # 使用模型计算输出结果

        target = target.float().squeeze() # 将标签做成一维张量
        output = output.squeeze()
        # loss = loss_func(output, target, net, sample_weights)
        # 根据损失函数的类型决定是否使用 sample_weights
        if isinstance(loss_func, RCELoss):
            loss = loss_func(output, target, net)  # 不使用样本权重
        else:
            loss = loss_func(output, target, net, sample_weights)  # 使用样本权重计算损失

        loss.backward()
        optimizer.step()
        # # 更新学习率
        # scheduler.step()

        total_loss += loss.item()
        prediction = (output > 0.5).int() # 如果>0.5，则相应位置的元素将变为整数1，否则为整数0
        correct += (prediction == target).sum().item()
        sample_sum += len(prediction)
        train_pred.extend(prediction.tolist()) # .tolist()转换为列表
        train_target.extend(target.tolist()) # .extend 将另一个列表中的元素添加到当前列表的末尾。

    loss = total_loss / train_batch_num
    acc = correct / sample_sum
    f_measure = f1_score(train_target, train_pred)
    auc = roc_auc_score(train_target, train_pred)
    p = precision_score(train_target, train_pred)
    r = recall_score(train_target, train_pred)

    return loss, acc, f_measure, auc, p, r


def test_epoch(net, data_loader, loss_func, optimizer):
    net.eval() # 设置为测试模式
    test_batch_num = len(data_loader)
    total_loss = 0
    correct = 0
    sample_num = 0
    test_pred, test_target = [], []

    # 不进行梯度变化
    with torch.no_grad():
        for batch_idx, (data, target, sample_weights) in enumerate(data_loader):
            data = data.float()
            target = target.long()
            output = net(data)

            target = target.float().squeeze()  # 将标签做成一维张量
            output = output.squeeze()
            # loss = loss_func(output, target, net)
            # 根据损失函数的类型决定是否使用 sample_weights
            if isinstance(loss_func, RCELoss):
                loss = loss_func(output, target, net)  # 不使用样本权重
            else:
                loss = loss_func(output, target, net, sample_weights)  # 使用样本权重计算损失

            total_loss += loss.item()
            prediction = (output > 0.5).int()
            correct += (prediction == target).sum().item()
            sample_num += len(prediction)
            test_pred.extend(prediction.tolist())
            test_target.extend(target.tolist())

    loss = total_loss / test_batch_num
    acc = correct / sample_num
    cm = confusion_matrix(test_target, test_pred)
    TN, FP, FN, TP = cm.flatten() # 混淆矩阵展平
    a, b, c, d = TP + FP, TP + FN, TN + FP, TN + FN
    if a * b * c * d != 0:
        MCC = (TP * TN - FP * FN) / math.sqrt(a * b * c * d)
    else:
        MCC = 0
    TPR = TP / (TP + FN)
    TNR = TN / (FP + TN)
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    G_mean = math.sqrt(TPR * TNR)
    F_measure = f1_score(test_target, test_pred)
    auc = roc_auc_score(test_target, test_pred)
    p = precision_score(test_target, test_pred)
    r = recall_score(test_target, test_pred)

    return test_target, test_pred, loss, acc, F_measure, G_mean, MCC, auc, FPR, FNR


def draw_loss(train_loss, test_loss):
    x = np.linspace(0, len(train_loss), len(train_loss))
    plt.plot(x, train_loss, label='Train_loss', linewidth=1.5)
    plt.plot(x, test_loss, label='Test_loss', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def draw_acc(train_acc, test_acc):
    x = np.linspace(0, len(train_acc), len(train_acc))
    plt.plot(x, train_acc, label='Train_acc', linewidth=1.5)
    plt.plot(x, test_acc, label='Test_acc', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def float_to_bin(number):
    """ Convert a float into binary using IEEE 754 format. """
    if number == 0.0:
        return "0" * 32
    packed = np.float32(number).tobytes()
    unpacked = np.frombuffer(packed, dtype=np.uint32)[0]
    return format(unpacked, '032b')

# 原始但有问题
# def bitwise_or(bin1, bin2):
#     """Perform bitwise OR operation between two binary strings"""
#     num1 = int(bin1, 2)
#     num2 = int(bin2, 2)
#     result = num1 | num2
#     # Return result as a binary string, formatted to maintain length
#     return format(result, '032b')

# 新的，正确
def bitwise_or(bin1, bin2):
    """Perform bitwise OR operation between two binary strings"""
    max_len = max(len(bin1), len(bin2))
    num1 = int(bin1, 2)
    num2 = int(bin2, 2)
    result = num1 | num2
    # Return result as a binary string, formatted to maintain length
    return format(result, f'0{max_len}b')

# 做成图像
def Padding_data(X, y):
    # 将二进制转化为数字并重塑为46x46的数组
    X_images = np.array([np.array(list(map(int, list(x)))).reshape(sizes, sizes) for x in X])
    X_images = X_images[:, None, :, :]  # 增加一个维度，以匹配CNN输入 (N, C, H, W)
    y = y.astype(np.int64)
    return torch.tensor(X_images, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)


# # 取出一个batch中样本分别对应的权重
# class WeightedMNISTDataset(torch.utils.data.Dataset):
#     def __init__(self, X_train, y_train, weights):
#         self.X_train = X_train
#         self.y_train = y_train
#         self.weights = weights
#
#     def __len__(self):
#         return len(self.X_train)
#
#     def __getitem__(self, idx):
#         X_train, y_train = self.X_train[idx], self.y_train[idx]
#         weight = self.weights[idx]
#         return X_train, y_train, weight


def Sim_weight(X_train_bin_com, X_test_bin_com):
    # 计算样本间的柯氏复杂度
    kol_list = []
    for i in range(len(X_train_bin_com)):
        c_x = X_train_bin_com[i].count('1')
        for j in range(len(X_train_bin_com)):
            if i == j:  # 保证对角线为1
                k = 1
                kol_list.append(k)
            else:
                c_y = X_train_bin_com[j].count('1')
                result = bitwise_or(X_train_bin_com[i], X_train_bin_com[j])
                c_xy = result.count('1')
                k = (c_xy - min(c_x, c_y)) / max(c_x, c_y)
                kol_list.append(k)
    kol_matrix = 1 - (np.array(kol_list).reshape(X_train_num, X_train_num))  # 二维相似性矩阵->权重矩阵
    # kol_image = torch.tensor(kol_matrix, dtype=torch.float32)  # 相似性矩阵图像
    # kol_image = kol_image.unsqueeze(1)
    # # print(kol_image.shape)

    # 相似性矩阵获得的权重
    weights = np.sum(kol_matrix, axis=1) / X_train_num
    weights = torch.tensor(weights, dtype=torch.float32)

    # 计算测试集与训练集的样本相似性
    kol_test_list = []
    for i in range(len(X_test_bin_com)):
        c_x = X_test_bin_com[i].count('1')
        for j in range(len(X_train_bin_com)):
            c_y = X_train_bin_com[j].count('1')
            result = bitwise_or(X_test_bin_com[i], X_train_bin_com[j])
            c_xy = result.count('1')
            k = (c_xy - min(c_x, c_y)) / max(c_x, c_y)
            kol_test_list.append(k)
    kol_test_matrix = 1 - (np.array(kol_test_list).reshape(X_test_num, X_train_num))
    # kol_test_image = torch.tensor(kol_matrix, dtype=torch.float32)
    # kol_test_image = kol_test_image.unsqueeze(1)

    # test_weights = np.sum(kol_matrix, axis=1) / X_test_num
    test_weights = np.sum(kol_test_matrix, axis=1) / X_test_num
    test_weights = torch.tensor(test_weights, dtype=torch.float32)

    return weights, test_weights


if __name__ == '__main__':

    start_time = time.time()

    data_frame = np.array(pd.read_csv(r'D:\A论文实验\data\JIRA\groovy-1_6_BETA_1.csv')) # ......................................
    print('样本个数: ', data_frame.shape[0], '特征个数: ', data_frame.shape[1] - 1)
    data = data_frame[:, : -1]
    target = data_frame[:, -1]
    print('缺陷率：', np.sum(target == 1) / data.shape[0])

    # 归一化
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    sizes = int(math.sqrt(32 * data.shape[1])) + 1 # 实力图像尺寸 65-->46, 36
    zero_pad = sizes * sizes - 32 * data.shape[1] # 填补0的个数
    print('尺寸：%d, 填补数量：%d'%(sizes, zero_pad))
    n_split = 5
    kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)

    Acc_list, F_measure_list, G_mean_list, MCC_list, AUC_list, P_list, R_list, FPR_list, FNR_list = [], [], [], [], [], [], [], [], []
    F_measure_cnnlist, G_mean_cnnlist, MCC_cnnlist, AUC_cnnlist, P_cnnlist, R_cnnlist, FPR_cnnlist, FNR_cnnlist = [], [], [], [], [], [], [], []
    F_measure_list1, G_mean_list1, MCC_list1, AUC_list1, P_list1, R_list1, FPR_list1, FNR_list1 = [], [], [], [], [], [], [], []
    train_loss_aver, test_loss_aver, train_acc_aver, test_acc_aver = [], [], [], []
    for kf, (train_index, test_index) in enumerate(kfold.split(data)):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]
        # 平衡数据集
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X_train, y_train)
        # print(X_train.shape)
        X_train_num, X_test_num = X_train.shape[0], X_test.shape[0]
        # 转换二进制
        X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test)
        X_train_bin, X_test_bin = X_train.applymap(float_to_bin), X_test.applymap(float_to_bin)
        # print(X_train_bin)
        # 样本二进制特征值组合
        X_train_bin_com, X_test_bin_com = X_train_bin.apply(lambda row: ''.join(row), axis=1), X_test_bin.apply(lambda row: ''.join(row), axis=1)
        # print(X_train_bin_com.apply(len))
        # 二进制填充零
        X_train_bin_com, X_test_bin_com = X_train_bin_com.apply(lambda x: x + '0' * zero_pad), X_test_bin_com.apply(lambda x: x + '0' * zero_pad)

        # 计算相似性矩阵获得的权重
        weights, test_weights = Sim_weight(X_train_bin_com, X_test_bin_com)

        # 做成实例图像
        X_train_bin_com, y_train = Padding_data(X_train_bin_com, y_train)
        X_test_bin_com, y_test = Padding_data(X_test_bin_com, y_test)

        batch_size = 64 # .........................................................................
        lr = 1e-5
        epochs = 20
        # 创建TensorDataset
        train_dataset = TensorDataset(X_train_bin_com, y_train, weights)
        test_dataset = TensorDataset(X_test_bin_com, y_test, test_weights)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        net = ConvModule()
        active = NormalizedWeightedBCEWithLogitsLoss()
        passive = RCELoss()
        # criterion = ISL(weights=weights)
        optimizer = optim.Adam(net.parameters(), lr=lr)

        # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9) # 动量为0.9的SGD优化器
        # # 定义余弦学习率退火调度器
        # scheduler = CosineAnnealingLR(optimizer, T_max=20 eta_min=0)

        train_loss_list, train_acc_list, test_loss_list, test_acc_list = [], [], [], []
        test_f1_list, test_G_mean_list, test_MCC_list, test_auc_list, test_fpr_list, test_fnr_list = [], [], [], [], [], []
        train_time, test_time = 0, 0
        print('第%d轮' % (kf + 1), '*' * 50)

        for epoch in range(epochs):
            start_train_time = time.time()
            criterion = ActiveNegativeLoss(active_loss=active, negative_loss=passive)
            train_loss, train_acc, train_f_measure, train_auc, train_p, train_r = train_epoch(net, data_loader=train_loader, loss_func=criterion, optimizer=optimizer)
            end_train_time = time.time()
            train_time += (end_train_time - start_train_time)

            start_test_time = time.time()
            criterion = ActiveNegativeLoss(active_loss=active, negative_loss=passive)
            net_target, net_pred, test_loss, test_acc, test_f1, test_G_mean, test_MCC, test_auc, test_fpr, test_fnr = test_epoch(net, data_loader=test_loader, loss_func=criterion, optimizer=optimizer)
            end_test_time = time.time()
            test_time += (end_test_time - start_test_time)

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

            # 存储每次循环中各个评价指标值，方便统计十折之后的平均评价指标值
            test_f1_list.append(test_f1)
            test_G_mean_list.append(test_G_mean)
            test_MCC_list.append(test_MCC)
            test_auc_list.append(test_auc)
            test_fpr_list.append(test_fpr)
            test_fnr_list.append(test_fnr)

            print(f' Epoch [{epoch + 1}/{epochs}] train_loss:{train_loss:.4f}'
                  f' train_acc:{train_acc:.4f} train_auc:{train_auc:.4f}\n'
                  f' test_loss:{test_loss:.4f} test_acc:{test_acc:.4f}'
                  f' F_measure:{test_f1:.4f} G_mean:{test_G_mean:.4f} MCC:{test_MCC:.4f}'
                  f' AUC:{test_auc:.4f} FPR:{test_fpr:.4f} FNR:{test_fnr:.4f}')

        Acc_list.append(test_acc_list[-1])
        F_measure_list.append(test_f1_list[-1])
        G_mean_list.append(test_G_mean_list[-1])
        MCC_list.append(test_MCC_list[-1])
        AUC_list.append(test_auc_list[-1])
        FPR_list.append(test_fpr_list[-1])
        FNR_list.append(test_fnr_list[-1])

        train_loss_aver.append(train_loss_list)
        test_loss_aver.append(test_loss_list)
        train_acc_aver.append(train_acc_list)
        test_acc_aver.append(test_acc_list)

        print('*' * 50)
        print('训练总时间:%.4f, 测试总时间:%.4f' % (train_time, test_time))

    # 计算每个epoch的平均损失
    average_train_loss = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in zip(*train_loss_aver)]
    average_test_loss = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in zip(*test_loss_aver)]
    average_train_acc = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in zip(*train_acc_aver)]
    average_test_acc = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in zip(*test_acc_aver)]

    data_save = {
        'Epoch': list(range(1, len(average_train_loss) + 1)),
        'Train Loss': average_train_loss,
        'Test Loss': average_test_loss,
        'Train Accuracy': average_train_acc,
        'Test Accuracy': average_test_acc,
    }
    data_save = pd.DataFrame(data_save)
    output_file = r'C:\Users\Thinker\Desktop\Kolmogorov\实验\loss data\hive.xlsx' # ......................................
    data_save.to_excel(output_file, index=False)

    # draw_loss(average_train_loss, average_test_loss)
    # draw_acc(average_train_acc, average_test_acc)

    print(Acc_list)
    print(F_measure_list)
    print(G_mean_list)
    print(MCC_list)
    print(AUC_list)
    print(FPR_list)
    print(FNR_list)

    print('*' * 50)
    print('ISRL Accuracy:%.4f+%.4f' % (np.mean(Acc_list), np.std(Acc_list)))
    print('ISRL F-measure:%.4f+%.4f'%(np.mean(F_measure_list), np.std(F_measure_list)))
    print('ISRL G-mean:%.4f+%.4f'%(np.mean(G_mean_list), np.std(G_mean_list)))
    print('ISRL MCC:%.4f+%.4f'%(np.mean(MCC_list), np.std(MCC_list)))
    print('ISRL AUC:%.4f+%.4f'%(np.mean(AUC_list), np.std(AUC_list)))
    print('ISRL FPR:%.4f+%.4f'%(np.mean(FPR_list), np.std(FPR_list)))
    print('ISRL FNR:%.4f+%.4f'%(np.mean(FNR_list), np.std(FNR_list)))

    end_time = time.time()
    print('总时间:', (end_time - start_time))

# hive
# [0.8133802816901409, 0.8586572438162544, 0.8409893992932862, 0.872791519434629, 0.8374558303886925]
# [0.576, 0.6875, 0.6762589928057554, 0.6785714285714285, 0.6166666666666668]
# [0.8007577704536543, 0.8244441277959138, 0.8141870826312319, 0.7613300287263668, 0.7579214747996181]
# [0.49491149846094695, 0.6035996309114351, 0.5805568715024223, 0.6058059208735739, 0.514890851817989]
# [0.8009682133723054, 0.8262303990063654, 0.8154260818195245, 0.7793022916362573, 0.7670392796149668]
# [0.18067226890756302, 0.11946902654867257, 0.13963963963963963, 0.05429864253393665, 0.11504424778761062]
# [0.21739130434782608, 0.22807017543859648, 0.22950819672131148, 0.3870967741935484, 0.3508771929824561]
# **************************************************
# ISRL Accuracy:0.8447+0.0202
# ISRL F-measure:0.6470+0.0435
# ISRL G-mean:0.7917+0.0273
# ISRL MCC:0.5600+0.0462
# ISRL AUC:0.7978+0.0220
# ISRL FPR:0.1218+0.0410
# ISRL FNR:0.2826+0.0716
# 总时间: 5688.281777620316
