import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import auc

# 使用TkAgg作为后端
matplotlib.use('TkAgg')

# 定义要读取的CSV文件列表
csv_files = ['Result/aupr_data2.csv', 'Result/aupr_data3.csv', 'Result/aupr_data4.csv', 'Result/aupr_data6.csv', 'Result/aupr_data8.csv']
labels = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']  # 每个模型的标签

plt.figure(figsize=(8, 6))

# 存储每个模型的Recall和Precision
all_recall = []
all_precision = []
aupr = [96.08, 95.07, 96.10, 96.00,95.61]

# 遍历每个CSV文件，绘制曲线
for i, csv_file in enumerate(csv_files):
    # 读取CSV文件
    data = pd.read_csv(csv_file)
    recall = data['Recall'].values
    precision = data['Precision'].values

    # 绘制AUPR曲线
    plt.plot(recall, precision,  label=f'{labels[i]} (AUPR = {aupr[i]:.2f})', linewidth=2)  # 这里的linewidth设置为0.5

    # 存储每个模型的Recall和Precision
    all_recall.append(recall)
    all_precision.append(precision)

# 计算平均AUPR曲线
max_length = max(len(recall) for recall in all_recall)
mean_precision = np.zeros(max_length)

# 对每个Recall位置取平均
for recall in all_recall:
    interp_precision = np.interp(np.linspace(0, 1, max_length), recall, np.interp(recall, recall, precision))
    mean_precision += interp_precision

mean_precision /= len(csv_files)  # 计算平均

# 对角线
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")  # 设置对角线宽度

# 绘制平均AUPR曲线
plt.plot(np.linspace(0, 1, max_length), mean_precision, lw=2, color='gray', label='Mean AUPR:95.77', linestyle='--')  # 这里的linewidth设置为0.5

# 设置图形属性
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR curves')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid()
plt.legend()

plt.show()
