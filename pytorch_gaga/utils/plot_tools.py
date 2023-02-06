import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def plot_roc_curve(y_true, y_prob, name='ROC-AUC Curve'):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name)
    # display.plot()
    # figure = display.figure_

    fig = plt.figure()
    plt.title(name)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('FPR', fontsize=14)
    plt.ylabel('TPR', fontsize=14)
    plt.tick_params(labelsize=10)

    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")

    plt.legend(loc='lower right')

    # plt.show()

    return fig


def plot_pr_curve(y_true, y_prob, name='PR Curve'):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_prob)
    ap_gnn = metrics.average_precision_score(y_true, y_prob)
    # display = metrics.PrecisionRecallDisplay(precision, recall, estimator_name=name)
    # display.plot()
    # figure = display.figure_

    fig = plt.figure()
    plt.title(name)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('FPR', fontsize=14)
    plt.ylabel('TPR', fontsize=14)
    plt.tick_params(labelsize=10)

    plt.plot(recall, precision, label=f"AP={ap_gnn:.4f}")

    plt.legend(loc='lower left')

    # plt.show()

    return fig
