# ！C:\Users\liu\.conda\envs\liuenv python3.9
# -*- coding: utf-8 -*-
"""
@FileName: 单层感知机  -
@Author: liuzb
@Email: liuzhaobang@mail.ustc.edu.cn
@Date: 2024/8/1
@Description: 
"""
import numpy as np
import matplotlib.pyplot as plt


def perceptron(X, y, lr=0.1, n_iter=100):
    # 获取样本数量和特征数量
    n_samples, n_features = X.shape

    # 初始化权重和偏置为0
    weights = np.zeros(n_features)
    bias = 0.0

    # 记录误分类点以及迭代次数
    mis_points = []
    for _ in range(n_iter):
        # 标记是否有误分类点
        mis_flag = False

        for i in range(n_samples):
            # 计算预测值
            y_pred = np.dot(weights, X[i]) + bias

            # 根据预测值和真实值调整权重和偏置
            if y_pred * y[i] <= 0:
                # 根据误差进行权重和偏置的更新
                weights += lr * y[i] * X[i]
                bias += lr * y[i]

                # 标记有误分类点
                mis_flag = True
                mis_points.append((X[i, 0], X[i, 1], y[i]))

        # 如果没有误分类点，则提前结束迭代
        if not mis_flag:
            break

    # 返回训练好的权重和偏置以及误分类点
    return weights, bias, mis_points


# 生成数据集
np.random.seed(0)
X = np.random.randn(100, 2)
y = np.array([1 if x1 + x2 >= 0 else -1 for x1, x2 in X])

# 训练模型
weights, bias, mis_points = perceptron(X, y)

# 绘制决策边界和误分类点
x1 = np.arange(-3, 3, 0.1)
x2 = (-weights[0] * x1 - bias) / weights[1]
plt.plot(x1, x2, 'r', label='Decision Boundary')
for x, y, label in mis_points:
    if label == 1:
        plt.plot(x, y, 'bo')
    else:
        plt.plot(x, y, 'ro')

plt.legend()
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import time
#
#
# def perceptron(X, y, lr=0.1, n_iter=100):
#     # 获取样本数量和特征数量
#     n_samples, n_features = X.shape
#
#     # 初始化权重和偏置为0
#     weights = np.zeros(n_features)
#     bias = 0.0
#
#     # 记录误分类点以及迭代次数
#     mis_points = []
#     for _ in range(n_iter):
#         # 标记是否有误分类点
#         mis_flag = False
#
#         for i in range(n_samples):
#             # 计算预测值
#             y_pred = np.dot(weights, X[i]) + bias
#
#             # 根据预测值和真实值调整权重和偏置
#             if y_pred * y[i] <= 0:
#                 # 根据误差进行权重和偏置的更新
#                 weights += lr * y[i] * X[i]
#                 bias += lr * y[i]
#
#                 # 标记有误分类点
#                 mis_flag = True
#                 mis_points.append((X[i, 0], X[i, 1], y[i]))
#
#         plt.pause(0.1)
#         # 绘制决策边界和误分类点
#         x1 = np.arange(-3, 3, 0.1)
#         x2 = (-weights[0] * x1 - bias) / weights[1]
#         plt.plot(x1, x2, 'r-', label='Decision Boundary')
#
#         # 如果没有误分类点，则提前结束迭代
#         if not mis_flag:
#             break
#
#     # 返回训练好的权重和偏置以及误分类点
#     return weights, bias, mis_points
#
#
# # 生成数据集
# np.random.seed(0)
# # 生成一个100行2列的随机数矩阵，服从标准正态分布（均值为0，方差为1）。也可以理解为在二维空间中随机生成100个点。
# # 这是用于训练单层感知机的样本特征矩阵，每行代表一个样本的特征，每列代表一种特征。
# X = np.random.randn(100, 2)
# # 根据输入数据 X 生成对应的标签 y。其中 x1 和 x2 分别表示样本的两个特征，这里的目的是为了生成一个二分类任务的标签，使得单层感知机可以通过学习来对样本进行分类。
# y = np.array([1 if x1 + x2 >= 0 else -1 for x1, x2 in X])
#
# # 训练模型
# weights, bias, mis_points = perceptron(X, y)
#
# # 绘制决策边界和误分类点
# x1 = np.arange(-3, 3, 0.1)
# x2 = (-weights[0] * x1 - bias) / weights[1]
# plt.plot(x1, x2, 'r-', label='Decision Boundary')
# for x, y, label in mis_points:
#     if label == 1:
#         plt.plot(x, y, 'bo')
#     else:
#         plt.plot(x, y, 'ro')
#
# plt.legend()
# plt.show()

#
# for i, j, label in mis_points:
#     plt.pause(0.1)
#     #time.sleep(0.1)
#     if label == 1:
#         plt.plot(i, j, 'bo', markeredgewidth=6, markeredgecolor="grey")
#     else:
#         plt.plot(i, j, 'ro', markeredgewidth=6, markeredgecolor="grey")
#
# for i, v in enumerate(X):
#     #time.sleep(0.1)
#     plt.pause(0.1)
#     if y[i] == 1:
#         plt.plot(v[0], v[1], 'bo')
#     else:
#         plt.plot(v[0], v[1], 'ro')
