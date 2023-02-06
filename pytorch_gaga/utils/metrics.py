import torch
import os
from sklearn import metrics
import numpy as np
from collections import namedtuple


def calc_acc(y_true, y_pred):
    """
    Compute the accuracy of prediction given the labels.
    """
    # return (y_pred == y_true).sum() * 1.0 / len(y_pred)
    return metrics.accuracy_score(y_true, y_pred)


def calc_f1(y_true, y_pred):
    f1_binary_1_gnn = metrics.f1_score(y_true, y_pred, pos_label=1, average='binary')
    f1_binary_0_gnn = metrics.f1_score(y_true, y_pred, pos_label=0, average='binary')
    f1_micro_gnn = metrics.f1_score(y_true, y_pred, average='micro')
    f1_macro_gnn = metrics.f1_score(y_true, y_pred, average='macro')

    return f1_binary_1_gnn, f1_binary_0_gnn, f1_micro_gnn, f1_macro_gnn


def calc_roc_and_thres(y_true, y_prob):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob)
    auc_gnn = metrics.auc(fpr, tpr)

    # 约登指数求auc曲线下的最佳thres
    J = tpr - fpr
    # 输出KS值
    ks_val = max(abs(J))
    print(f"(Kolmogorov-Smirnov) KS = {ks_val:>2.4f}\n")
    
    idx = J.argmax(axis=0)
    best_thres = thresholds[idx]
    return auc_gnn, best_thres


def calc_ap_and_thres(y_true, y_prob):
    # \\text{AP} = \\sum_n (R_n - R_{n-1}) P_n, 和AUPRC略有不同
    ap_gnn = metrics.average_precision_score(y_true, y_prob)

    # 计算最大的F1
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_prob)
    F1 = 2 * precision * recall / (precision + recall)
    idx = F1.argmax(axis=0)
    best_thres = thresholds[idx]

    return ap_gnn, best_thres


def calc_gmean(conf):
    tn, fp, fn, tp = conf.ravel()
    return (tp * tn / ((tp + fn) * (tn + fp))) ** 0.5


def prob2pred(y_prob, thres=0.5):
    """
    Convert probability to predicted results according to given threshold
    :param y_prob: numpy array of probability in [0, 1]
    :param thres: binary classification threshold, default 0.5
    :returns: the predicted result with the same shape as y_prob
    """
    y_pred = np.zeros_like(y_prob, dtype=np.int32)
    y_pred[y_prob >= thres] = 1
    y_pred[y_prob < thres] = 0
    return y_pred


def to_numpy(x):
    # Convert tensor on the gpu to tensor of the cpu.
    if isinstance(x, torch.autograd.Variable):
        x = x.data
    return x.cpu().numpy() if x.is_cuda else x.numpy()


def convert_probs(labels, probs, threshold_moving=True, thres=0.5):
    labels = to_numpy(labels)
    probs = torch.nn.Sigmoid()(probs)
    probs = to_numpy(probs)
    probs_1 = probs[:, 1]
    if threshold_moving:
        preds = prob2pred(probs_1, thres=thres)
    else:
        preds = probs.argmax(axis=1)

    # 这里probs_1可能存在一点问题
    return labels, probs_1, preds


def eval_model(y_true, y_prob, y_pred):
    """计算各评价指标
    :param y_true: torch.Tensor
    :param y_prob: torch.Tensor
    :param y_pred: torch.Tensor
    :return: namedtuple
    """
    # 下面这部分写得不好看, 存在较多类型转换和指标冗余计算
    acc = calc_acc(y_true, y_pred)

    # 计算几个F1
    f1_binary_1, f1_binary_0, f1_micro, f1_macro = calc_f1(y_true, y_pred)

    # 计算ROC-AUC并求最佳阈值
    auc_gnn, best_roc_thres = calc_roc_and_thres(y_true, y_prob)
    # auc_gnn = metrics.roc_auc_score(labels, probs_1)

    # 计算AUPRC并求最佳阈值
    ap_gnn, best_pr_thres = calc_ap_and_thres(y_true, y_prob)

    # recall的计算注意，有 binary+pos_label 的值，有 micro 和 macro 的值
    precision_1 = metrics.precision_score(y_true, y_pred, pos_label=1, average="binary")
    recall_1 = metrics.recall_score(y_true, y_pred, pos_label=1, average='binary')
    recall_macro = metrics.recall_score(y_true, y_pred, average='macro')

    # 计算混淆矩阵(多余操作)
    conf_gnn = metrics.confusion_matrix(y_true, y_pred)
    gmean_gnn = calc_gmean(conf_gnn)
    tn, fp, fn, tp = conf_gnn.ravel()

    # 1:fraud->positive, 0:benign->negtive
    print(f"f1-macro={f1_macro:>2.4f} | AUC={auc_gnn:>2.4f}\n"
          f"Gmean={gmean_gnn:>2.4f} | AP(gnn)={ap_gnn:>2.4f}\n"
          f"Precision(1)={precision_1:>2.4f} | Recall(1)={recall_1:>2.4f}")

    print(f"TN={tn:>5d} FP={fp:>5d} FN={fn:>5d} TP={tp:>5d}")

    print(f"f1-fraud={f1_binary_1:>2.4f} | f1-benign={f1_binary_0:>2.4f}\n"
          f"f1-micro={f1_micro:>2.4f} | f1-macro={f1_macro:>2.4f}\n"
          f"ACC={acc:>2.4f} | Recall(macro)={recall_macro:>2.4f}\n")

    # print(metrics.classification_report(y_true=labels, y_pred=preds, digits=4))

    DataType = namedtuple('Metrics', ['f1_binary_1', 'f1_binary_0', 'f1_macro', 'auc_gnn',
                                      'gmean_gnn', 'recall_1', 'precision_1', 'ap_gnn',
                                      'best_roc_thres', 'best_pr_thres', 'recall_macro'])
    results = DataType(f1_binary_1=f1_binary_1, f1_binary_0=f1_binary_0, f1_macro=f1_macro,
                       auc_gnn=auc_gnn, gmean_gnn=gmean_gnn, ap_gnn=ap_gnn,
                       recall_1=recall_1, precision_1=precision_1, recall_macro=recall_macro,
                       best_pr_thres=best_pr_thres, best_roc_thres=best_roc_thres)

    return results


def calc_mean_sd(results):
    results = np.around(results, decimals=5)
    results = results[:, :-1]

    MEAN = np.mean(results, axis=0)
    # PSD = np.std(results, axis=0)
    SSD = np.std(results, axis=0, ddof=1)

    # print(MEAN, PSD, SSD)
    metric_name = ['f1_macro', 'auc', 'gmean',
                   'precision_1', 'recall_1', 'ap',
                   'f1_binary_1', 'f1_binary_0', 'recall_macro']
    for i, name in enumerate(metric_name):
        print("{}= {:1.4f}±{:1.4f}".format(name, MEAN[i], SSD[i]))


class ThresholdSelector:
    def __init__(self, dataset_name, start_wall_time, log_dir=None):
        self.save_dir = os.path.join('threshold_moving', dataset_name)
        if log_dir:
            self.save_dir = os.path.join(log_dir, self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        self.file_name = f"val_probs_{start_wall_time}.npy"
        self.save_path = os.path.join(self.save_dir, self.file_name)

        # probs: torch.Tensor
        self.probs = None
        self.best_thres = 0.5

        print(f"[{self.__class__.__name__}] Saving threshold_moving_logs to {self.save_path}")

    def save_probs(self, probs: torch.Tensor):
        self.probs = probs
        np.savez(self.save_path, probs=probs.cpu().numpy())

    def load_probs(self):
        data = np.load(self.save_path)
        self.probs = torch.tensor(data['probs'])
        return self.probs

    def threshold_moving(self, labels, thresholds):
        benchmark_list = []
        result_list = []
        for thres in thresholds:
            print(f"Thres={thres:.2f}")
            result = eval_model(labels, self.probs[:, 1], thres=thres)
            result_list.append(result)
            benchmark_list.append(result.f1_binary_0)

        self.best_thres = thresholds[np.array(benchmark_list).argmax(axis=0)]
        print(f"[{self.__class__.__name__}] Best threshold is {self.best_thres:.2f}")

        return self.best_thres
