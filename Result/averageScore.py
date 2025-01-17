
import pandas as pd
import numpy as np

# 定义5个CSV文件的路径
csv_files = [
    "prediction_fold_0.csv",
    "prediction_fold_1.csv",
    "prediction_fold_2.csv",
    "prediction_fold_3.csv",
    "prediction_fold_4.csv",
]

# 初始化累加矩阵
total_score = None

# 遍历每个文件，累加矩阵
for file in csv_files:
    score_matrix = pd.read_csv(file, header=None).values  # 读取CSV文件为NumPy数组
    if total_score is None:
        total_score = score_matrix  # 第一次初始化
    else:
        total_score += score_matrix  # 累加

# 求平均矩阵
average_score = total_score / len(csv_files)

# 保存最终的平均矩阵为Excel文件
average_score_df = pd.DataFrame(average_score)
average_score_df.to_excel("average_score.xlsx", index=False, header=False)

print("平均分数矩阵已保存为 average_score.xlsx")
