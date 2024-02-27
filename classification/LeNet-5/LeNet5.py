import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # ����һ�������ͳػ��㣬�ֱ��ӦLeNet5�е�C1��S2��
        # ����������ͨ��Ϊ1�����ͨ��Ϊ6�����þ���˴�С5x5������Ϊ1
        # �ػ����kernel��СΪ2x2
        self._conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        # ����һ�������ͳػ��㣬�ֱ��ӦLeNet5�е�C3��S4��
        # ����������ͨ��Ϊ6�����ͨ��Ϊ16�����þ���˴�С5x5������Ϊ1
        # �ػ����kernel��СΪ2x2
        self._conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        # ��ӦLeNet5��C5����㣬��������ȫ���Ӳ����ƣ���������ʹ����nn.Linearģ��
        # ����������ͨ����Ϊ4x4x16���������Ϊ120x1
        self._fc1 = nn.Sequential(
            nn.Linear(in_features=4*4*16, out_features=120)
        )
        # ��ӦLeNet5�е�F6��������120ά�����������84ά����
        self._fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84)
        )
        # ��ӦLeNet5�е�����㣬������84ά�����������10ά����
        self._fc3 = nn.Sequential(
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, input):
        # ǰ�򴫲�
        # MNIST DataSet image's format is 28x28x1
        # [28,28,1]--->[24,24,6]--->[12,12,6]
        conv1_output = self._conv1(input)
        # [12,12,6]--->[8,8,,16]--->[4,4,16]
        conv2_output = self._conv2(conv1_output)
        # ��[n,4,4,16]ά��ת��Ϊ[n,4*4*16]
        conv2_output = conv2_output.view(-1, 4 * 4 * 16)
        # [n,256]--->[n,120]
        fc1_output = self._fc1(conv2_output)
        # [n,120]-->[n,84]
        fc2_output = self._fc2(fc1_output)
        # [n,84]-->[n,10]
        fc3_output = self._fc3(fc2_output)
        return fc3_output