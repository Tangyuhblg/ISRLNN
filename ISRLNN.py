import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
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


# 检测GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')


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


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, reduction='mean', delta = 5e-5):
        """
        :param pos_weight: A weight of positive examples.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.register_buffer('pos_weight', pos_weight)
        self.reduction = reduction
        self.delta = delta

    def forward(self, input, target, model, sample_weights):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        # Calculate binary cross entropy with logits
        loss = F.binary_cross_entropy_with_logits(input, target, pos_weight=self.pos_weight, reduction='none')

        # Apply the sample weights
        if sample_weights is not None:
            loss = loss * sample_weights
        l1_norm = sum(p.abs().sum() for p in model.parameters())

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean() + self.delta * l1_norm
        elif self.reduction == 'sum':
            return loss.sum() + self.delta * l1_norm
        else:
            return loss + self.delta * l1_norm
#
# # 归一化的ISL
# class NormalizedWeightedBCEWithLogitsLoss(nn.Module):
#     def __init__(self, pos_weight=None, reduction='mean'):
#         """
#         :param pos_weight: 正样本的权重。
#         :param reduction: 指定对输出应用的归约方式：'none' | 'mean' | 'sum'。
#         """
#         super(NormalizedWeightedBCEWithLogitsLoss, self).__init__()
#         self.register_buffer('pos_weight', pos_weight)
#         self.reduction = reduction
#
#     def forward(self, input, target, sample_weights=None):
#         # 确保输入和目标张量的大小一致
#         if not (target.size() == input.size()):
#             raise ValueError("目标大小 ({}) 必须与输入大小 ({}) 一致".format(target.size(), input.size()))
#
#         # 计算二元交叉熵损失（不应用归约）
#         loss = F.binary_cross_entropy_with_logits(input, target, pos_weight=self.pos_weight, reduction='none')
#
#         # 如果提供了样本权重，则将其应用到损失上
#         if sample_weights is not None:
#             loss = loss * sample_weights
#
#         # 计算归一化项，使用logsigmoid函数处理输入
#         normalization_term = -torch.sum(F.logsigmoid(input), dim=-1)
#
#         # 将损失值进行归一化处理
#         normalized_loss = loss / normalization_term
#
#         # 根据指定的归约方式，返回归一化后的损失值
#         if self.reduction == 'mean':
#             return normalized_loss.mean()  # 返回均值
#         elif self.reduction == 'sum':
#             return normalized_loss.sum()  # 返回总和
#         else:
#             return normalized_loss  # 如果不指定归约方式，直接返回计算结果

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


# 注意力机制
class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=1, kernel_size=1) # 1x1卷积核用于学习注意力机制权重
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        attention_weight = self.conv1(x) # 计算注意力权重
        attention_weight = self.softmax(attention_weight) # 归一化
        attention_feature = x * attention_weight # 特征加权求和

        return attention_feature


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
        # 将模型移动到设备
        self.to(device)

    def forward(self, x):
        x = x.to(device)  # 确保输入在正确的设备上
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
        data = data.float().to(device)
        target = target.long().to(device)
        sample_weights = sample_weights.to(device) if sample_weights is not None else None

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
    # print(train_target)
    # print(train_pred)
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
            data = data.float().to(device)
            target = target.long().to(device)
            sample_weights = sample_weights.to(device) if sample_weights is not None else None

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

def bitwise_or(bin1, bin2):
    """Perform bitwise OR operation between two binary strings"""
    num1 = int(bin1, 2)
    num2 = int(bin2, 2)
    result = num1 | num2
    # Return result as a binary string, formatted to maintain length
    return format(result, '032b')

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


# Early stopping implementation
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, loss, model):
        if self.best_loss is None:
            self.best_loss = loss
            self.best_state_dict = model.state_dict()
        elif loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_state_dict(self.best_state_dict)  # Restore the best model
        else:
            self.best_loss = loss
            self.best_state_dict = model.state_dict()
            self.counter = 0


if __name__ == '__main__':

    start_time = time.time()

    data_frame = np.array(pd.read_csv(r'./data/jetty4.csv'))
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

    # # 早停机制
    # patience = 5  # Set patience for early stopping
    # early_stopping = EarlyStopping(patience=patience)

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
        # print(weights)

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
        net = net.to(device)
        active = WeightedBCEWithLogitsLoss()
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
            # print('训练时间:%.4f, 测试时间:%.4f' % ((end_train_time - start_train_time), (end_test_time - start_test_time)))

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

        torch.cuda.empty_cache() # 在每个fold结束后清理GPU缓存

        print('*' * 50)
        print('训练总时间:%.4f, 测试总时间:%.4f' % (train_time, test_time))

    # 计算每个epoch的平均损失
    average_train_loss = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in zip(*train_loss_aver)]
    average_test_loss = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in zip(*test_loss_aver)]
    average_train_acc = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in zip(*train_acc_aver)]
    average_test_acc = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in zip(*test_acc_aver)]
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

# gro
# [0.47058823529411764, 0.5116279069767442, 0.4166666666666667, 0.47619047619047616, 0.4615384615384615]
# [0.88995662367488, 0.7433111162394346, 0.729448143869456, 0.8200244768534768, 0.7578164119928482]
# [0.4939654152891997, 0.44805216423119265, 0.35666155265876803, 0.4555932883365403, 0.41682267719916005]
# [0.8899572649572649, 0.7576103500761036, 0.7381756756756757, 0.8217014773306164, 0.768095238095238]
# **************************************************
# DCNN+WStacking F-measure:0.4673+0.0305
# DCNN+WStacking G-mean:0.7881+0.0596
# DCNN+WStacking MCC:0.4342+0.0459
# DCNN+WStacking AUC:0.7951+0.0549
# DCNN+WStacking FPR:0.1172+0.0184
# DCNN+WStacking FNR:0.2926+0.1068

# LC
# [0.8273381294964028, 0.8768115942028986, 0.8478260869565217, 0.7536231884057971, 0.8478260869565217]
# [0.47826086956521735, 0.0, 0.27586206896551724, 0.19047619047619047, 0.4878048780487804]
# [0.7842536346366908, 0.0, 0.7020766827663397, 0.5590169943749475, 0.7615417253558695]
# [0.4263756860322179, 0, 0.26017051055834156, 0.11134098992615366, 0.42711155806655376]
# [0.7860215053763441, 0.5, 0.7170119956379498, 0.590625, 0.7682926829268292]
# [0.16129032258064516, 0.0, 0.13740458015267176, 0.21875, 0.13008130081300814]
# [0.26666666666666666, 1.0, 0.42857142857142855, 0.6, 0.3333333333333333]
# **************************************************
# ISRL F-measure:0.8307+0.0416
# ISRL F-measure:0.2865+0.1837
# ISRL G-mean:0.5614+0.2914
# ISRL MCC:0.2450+0.1698
# ISRL AUC:0.6724+0.1100
# ISRL FPR:0.1295+0.0718
# ISRL FNR:0.5257+0.2623
# 总时间: 3244.4909212589264

# act
# [0.8753315649867374, 0.8912466843501327, 0.8249336870026526, 0.8779840848806366, 0.8856382978723404]
# [0.6178861788617885, 0.6870229007633588, 0.5714285714285714, 0.603448275862069, 0.6260869565217392]
# [0.7442538129020028, 0.847306102644698, 0.7706317006696777, 0.7591600539668435, 0.7902969012502252]
# [0.5441902269329277, 0.6303160918117241, 0.4783998296130208, 0.5318909221799257, 0.5614666958994107]
# [0.7633286741214057, 0.8494243421052632, 0.7743655848751391, 0.7735591900311527, 0.799374963490858]
# [0.0670926517571885, 0.090625, 0.14968152866242038, 0.0778816199376947, 0.0804953560371517]
# [0.40625, 0.21052631578947367, 0.30158730158730157, 0.375, 0.32075471698113206]
# **************************************************
# ISRL F-measure:0.8710+0.0237
# ISRL F-measure:0.6212+0.0378
# ISRL G-mean:0.7823+0.0358
# ISRL MCC:0.5493+0.0491
# ISRL AUC:0.7920+0.0311
# ISRL FPR:0.0932+0.0292
# ISRL FNR:0.3228+0.0675