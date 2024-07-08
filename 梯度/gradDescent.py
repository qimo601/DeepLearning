import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

'''
例：模拟梯度计算过程
函数的表达式是 f(x)=(x-1)**2+1
函数的导数是f'(x)=2x-2
更新x = x-eta*f'(x)
'''
# 初始算法开始之前的坐标
# cur_x 和 cur_y
cur_x = 6
cur_y = (cur_x - 1) ** 2 + 1
# 设置学习率 eta 为 0.05
eta = 0.05
# 变量 iter 用于存储迭代次数
# 这次我们迭代 1000 次
# 所以给它赋值 1000
iter = 1000
# 变量 cur_df 用于存储
# 当前位置的导数
# 一开始我给它赋值为 None
# 每一轮循环的时候为它更新值
cur_df = None

# all_x 用于存储
# 算法进行时所有点的横坐标
all_x = []
# all_y 用于存储
# 算法进行时所有点的纵坐标
all_y = []

# 把最一开始的坐标存储到
# all_x 和 all_y 中
all_x.append(cur_x)
all_y.append(cur_y)

a, b, c = 1, 2, 3


def f1():

    print(df)
    df.append(4)

def main():
    global cur_x
    # 循环结束也就意味着算法的结束
    for i in range(iter):
        # 每一次迭代之前先计算
        # 当前位置的梯度 cur_df
        # cur 是英文单词 current
        cur_df = 2 * cur_x - 2
        # 更新 cur_x 到下一个位置
        cur_x = cur_x - eta * cur_df
        # 更新下一个 cur_x 对应的 cur_y
        cur_y = (cur_x - 1) ** 2 + 1

        # 其实 cur_y 并没有起到实际的计算作用
        # 在这里计算 cur_y 只是为了将每一次的
        # 点的坐标存储到 all_x 和 all_y 中
        # all_x 存储了二维平面上所有点的横坐标
        # all_y 存储了二维平面上所欲点的纵坐标
        # 使用 list 的 append 方法添加元素
        all_x.append(cur_x)
        all_y.append(cur_y)

    # 这里的 x, y 值为了绘制二次函数
    # 的那根曲线用的，和算法没有关系
    # linspace 将会从区间 [-5, 7] 中
    # 等距离分割出 100 个点并返回一个
    # np.array 类型的对象给 x
    x = np.linspace(-5, 7, 100)
    # 计算出 x 中每一个横坐标对应的纵坐标
    y = (x - 1) ** 2 + 1
    # plot 函数会把传入的 x, y
    # 组成的每一个点依次连接成一个平滑的曲线
    # 这样就是我们看到的二次函数的曲线了
    plt.plot(x, y)
    # axis 函数用来指定坐标系的横轴纵轴的范围
    # 这样就表示了
    # 横轴为 [-7, 9]
    # 纵轴为 [0, 50]
    plt.axis([-7, 9, 0, 50])
    # scatter 函数是用来绘制散点图的
    # scatter 和 plot 函数不同
    # scatter 并不会将每个点依次连接
    # 而是直接将它们以点的形式绘制出来
    plt.scatter(np.array(all_x), np.array(all_y), color='red')
    plt.show()

def drawGif():
    #plt.style.use('seaborn-pastel')

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 4), ylim=(-2, 2))
    line, = ax.plot([], [], lw=3)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = np.linspace(0, 4, 1000)
        y = np.sin(2 * np.pi * (x - 0.01 * i))
        line.set_data(x, y)
        return line,

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=100, interval=20, blit=True)

    anim.save('sine_wave.gif', writer='pillow')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    drawGif()
