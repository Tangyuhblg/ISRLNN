'''Active Negative Loss Functions for Learning with
Noisy Labels'''
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import math
import warnings
warnings.filterwarnings('ignore')


class CrossEntropy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        ce = self.ce(pred, labels)
        return ce


class NormalizedNegativeCrossEntropy(torch.nn.Module): # passive loss
    def __init__(self, num_classes=2, min_prob=1e-7) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.min_prob = min_prob
        self.A = - torch.tensor(min_prob).log()

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = pred.clamp(min=self.min_prob, max=1)
        pred = self.A + pred.log()  # - log(1e-7) - (- log(p(k|x)))
        label_one_hot = F.one_hot(labels, self.num_classes).to(pred.device)
        nnce = 1 - (label_one_hot * pred).sum(dim=1) / pred.sum(dim=1)
        return nnce.mean()


class ActiveNegativeLoss(torch.nn.Module):
    def __init__(self, active_loss, negative_loss,
                 alpha=5., beta=5., delta=5e-5) -> None:
        super().__init__()
        self.active_loss = active_loss
        self.negative_loss = negative_loss
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def forward(self, pred, labels, model):
        al = self.active_loss(pred, labels)
        nl = self.negative_loss(pred, labels)
        l1_norm = sum(p.abs().sum() for p in model.parameters())

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

        return out


# 定义三层卷积，不使用池化，输入通道为1
class ConvModule(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvModule, self).__init__()
        drop_prob1, drop_prob2 = 0.2, 0.5 # 丢弃率

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

        self.fc = nn.Sequential(nn.Dropout(p=0.5),
                                 nn.Linear(64 * sizes * sizes, num_classes),  # 输出 num_classes
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

    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.float()
        target = target.long()
        optimizer.zero_grad()
        output = net(data) # 使用模型计算输出结果

        # target = target.float().squeeze() # 将标签做成一维张量
        output = output.squeeze()
        # print(output.shape, target.shape)
        loss = loss_func(output, target, net)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        prediction = output.argmax(dim=1) # num_class = 2
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
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.float()
            target = target.long()
            output = net(data)

            # target = target.float().squeeze() # 将标签做成一维张量
            output = output.squeeze()
            loss = loss_func(output, target, net)
            # print(loss.item())

            total_loss += loss.item()
            prediction = output.argmax(dim=1) # num_class = 2
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

if __name__ == '__main__':

    start_time = time.time()

    data_frame = np.array(pd.read_csv(r'D:\A论文实验\data\JIRA\groovy-1_6_BETA_1.csv'))
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
    train_loss_aver, test_loss_aver, train_acc_aver, test_acc_aver = [], [], [], []
    for kf, (train_index, test_index) in enumerate(kfold.split(data)):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]
        # 平衡数据集
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(X_train.shape)
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

        # 做成实例图像
        X_train_bin_com, y_train = Padding_data(X_train_bin_com, y_train)
        X_test_bin_com, y_test = Padding_data(X_test_bin_com, y_test)

        batch_size = 64
        lr = 1e-5
        epochs = 20
        # 创建TensorDataset
        train_dataset = TensorDataset(X_train_bin_com, y_train)
        test_dataset = TensorDataset(X_test_bin_com, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        net = ConvModule()
        # 主动损失
        active = torch.nn.CrossEntropyLoss()
        # 被动损失
        passive = NormalizedNegativeCrossEntropy()
        criterion = ActiveNegativeLoss(active_loss=active, negative_loss=passive)
        optimizer = optim.Adam(net.parameters(), lr=lr)

        train_loss_list, train_acc_list, test_loss_list, test_acc_list = [], [], [], []
        test_f1_list, test_G_mean_list, test_MCC_list, test_auc_list, test_fpr_list, test_fnr_list = [], [], [], [], [], []
        train_time, test_time = 0, 0
        print('第%d轮' % (kf + 1), '*' * 50)

        for epoch in range(epochs):
            start_train_time = time.time()
            train_loss, train_acc, train_f_measure, train_auc, train_p, train_r = train_epoch(net, data_loader=train_loader, loss_func=criterion, optimizer=optimizer)
            end_train_time = time.time()
            train_time += (end_train_time - start_train_time)

            start_test_time = time.time()
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
    output_file = r'C:\Users\Thinker\Desktop\Kolmogorov\实验\loss data\gro.xlsx'
    data_save.to_excel(output_file, index=False)

    print(Acc_list)
    print(F_measure_list)
    print(G_mean_list)
    print(MCC_list)
    print(AUC_list)
    print(FPR_list)
    print(FNR_list)

    print('*' * 50)
    print('ANL-CE Accuracy:%.4f+%.4f' % (np.mean(Acc_list), np.std(Acc_list)))
    print('ANL-CE F-measure:%.4f+%.4f' % (np.mean(F_measure_list), np.std(F_measure_list)))
    print('ANL-CE G-mean:%.4f+%.4f' % (np.mean(G_mean_list), np.std(G_mean_list)))
    print('ANL-CE MCC:%.4f+%.4f' % (np.mean(MCC_list), np.std(MCC_list)))
    print('ANL-CE AUC:%.4f+%.4f' % (np.mean(AUC_list), np.std(AUC_list)))
    print('ANL-CE FPR:%.4f+%.4f' % (np.mean(FPR_list), np.std(FPR_list)))
    print('ANL-CE FNR:%.4f+%.4f' % (np.mean(FNR_list), np.std(FNR_list)))

    end_time = time.time()
    print('总时间:', (end_time - start_time))

# [0.48275862068965514, 0.6153846153846153, 0.37837837837837834, 0.5945945945945945, 0.5625000000000001]
# [0.8443713418650368, 0.7909303232622376, 0.6293765684110466, 0.8793782826717071, 0.7773581634521595]
# [0.48319458967816775, 0.5659874192046348, 0.3045001522501142, 0.5809503726796124, 0.5210256331235729]
# [0.8472222222222222, 0.8025114155251141, 0.6714527027027026, 0.8800305654610291, 0.7914285714285713]
# **************************************************
# DCNN+WStacking F-measure:0.5267+0.0868
# DCNN+WStacking G-mean:0.7843+0.0857
# DCNN+WStacking MCC:0.4911+0.0995
# DCNN+WStacking AUC:0.7985+0.0710
# DCNN+WStacking FPR:0.0771+0.0138
# DCNN+WStacking FNR:0.3258+0.1395




