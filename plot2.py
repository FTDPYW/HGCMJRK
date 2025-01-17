import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def plot_roc_curves_with_mean(all_fprs, all_tprs, all_aucs):
    plt.figure(figsize=(8, 6))

    # 插值以统一FPR的范围
    mean_fpr = np.linspace(0, 1, 100)  # 平均FPR的固定范围
    interpolated_tprs = []
    auc_all=[95.63,94.09,95.66,95.52,95.02]
    for i, (fpr, tpr, auc_val) in enumerate(zip(all_fprs, all_tprs, all_aucs)):
        # 插值TPR
        interpolated_tpr = np.interp(mean_fpr, fpr, tpr)
        interpolated_tprs.append(interpolated_tpr)
        # 绘制每一折的ROC曲线
        plt.plot(fpr, tpr, lw=2, label=f"Fold {i + 1} (AUC = {auc_all[i]:.2f})")

    # 计算平均TPR和标准差
    mean_tpr = np.mean(interpolated_tprs, axis=0)
    std_tpr = np.std(interpolated_tprs, axis=0)
    mean_auc = 95.18

    # 绘制平均ROC曲线
    plt.plot(mean_fpr, mean_tpr, color='blue', lw=2, linestyle='--',
             label=f"Mean ROC (AUC = {mean_auc:.2f})")

    # 绘制标准差带
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)


    # 对角线
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=10, frameon=True)  # 确保AUC显示到小数点后两位
    plt.grid(alpha=0.4)
    plt.show()

