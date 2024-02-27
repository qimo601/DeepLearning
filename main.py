# This is a sample Python script.

# 胖子老板的父类
# 胖子老板的父类
class FatFather(object):
    def __init__(self, name):
        print('FatFather的init开始被调用')
        self.name = name
        print('调用FatFather类的name是%s' % self.name)
        print('FatFather的init调用结束')


# 胖子老板类 继承 FatFather 类
class FatBoss(FatFather):
    def __init__(self, name, hobby):
        print('胖子老板的类被调用啦！')
        self.hobby = hobby
        #FatFather.__init__(self,name)   # 直接调用父类的构造方法
        super().__init__(name)
        print("%s 的爱好是 %s" % (name, self.hobby))


def main():
    #ff = FatFather("胖子老板的父亲")
    fatboss = FatBoss("胖子老板", "打斗地主")




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
