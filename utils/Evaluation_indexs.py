from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import auc, roc_curve, roc_auc_score, precision_recall_curve
from math import sqrt

def evaluation_indexs(true_values, predict_values):
    cm = confusion_matrix(true_values, predict_values)
    TN, FP, FN, TP = confusion_matrix(true_values, predict_values).ravel()
    acc = metrics.accuracy_score(true_values, predict_values)
    p = metrics.precision_score(true_values, predict_values)
    r = metrics.recall_score(true_values, predict_values)
    F_measure = metrics.f1_score(true_values, predict_values)
    a = TP + FP
    b = TP + FN
    c = TN + FP
    d = TN + FN
    if a * b * c * d != 0:
        MCC = (TP * TN - FP * FN) / sqrt(a * b * c * d)
    else:
        MCC = 0
    TPR = TP / (TP + FN)
    TNR = TN / (FP + TN)
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    G_mean = sqrt(TPR * TNR)
    # 第三个参数pos_label*代表二进制数组y中代表“真”的二进制编码，其他的编码都会被认为是
    # fpr, tpr, thresholds = roc_curve(true_values, predict_values, pos_label=0) # 少数类label?
    auc_scores = metrics.roc_auc_score(true_values, predict_values)
    precision, recall, thresholds = precision_recall_curve(true_values, predict_values)
    aupr_scores = auc(recall, precision)

    return p, r, acc, F_measure, G_mean, MCC, auc_scores, aupr_scores, FPR, FNR