# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


class FatFather(object):
    def __init__(self, name):
        print('FatFather��init��ʼ������')
        self.name = name
        print('FatFather��name��%s' % self.name)
        print('FatFather��init���ý���')


def main():
    ff = FatFather("�����ϰ�ĸ���")


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
