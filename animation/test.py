# ！C:\Users\liu\.conda\envs\liuenv python3.9
# -*- coding: utf-8 -*-
"""
@FileName: test  -
@Author: liuzb
@Email: liuzhaobang@mail.ustc.edu.cn
@Date: 2024/10/30
@Description: 
"""
import numpy as np

# 生成一些示例数据
x = np.array([0, 1, 2, 3, 4])
y = np.array([1.1, 2.0, 2.9, 4.2, 5.1])

# 进行一次多项式拟合
coefficients = np.polyfit(x, y, 1)

# 打印拟合的系数
print(coefficients)