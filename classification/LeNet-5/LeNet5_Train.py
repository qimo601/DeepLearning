import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import matplotlib.pyplot as plt
from PyTorchVersion.Networks.LeNet5 import LeNet5

train_data = pd.DataFrame(pd.read_csv("../Data/mnist_train.csv"))

model = LeNet5()
print(model)

# ���彻������ʧ����
loss_fc = nn.CrossEntropyLoss()
# ��model�Ĳ�����ʼ��һ������ݶ��½��Ż���
optimizer = optim.SGD(params=model.parameters(),lr=0.001, momentum=0.78)
loss_list = []
x = []

# ��������1000��
for i in range(1000):
    # С�������ݼ���С����Ϊ30
    batch_data = train_data.sample(n=30, replace=False)
    # ÿһ�����ݵĵ�һ��ֵ�Ǳ�ǩ����
    batch_y = torch.from_numpy(batch_data.iloc[:,0].values).long()
    #ͼƬ��Ϣ��һ������784ά����ת��Ϊͨ����Ϊ1����С28*28��ͼƬ��
    batch_x = torch.from_numpy(batch_data.iloc[:,1::].values).float().view(-1,1,28,28)

    # ǰ�򴫲�����������
    prediction = model.forward(batch_x)
    # ������ʧֵ
    loss = loss_fc(prediction, batch_y)
    # Clears the gradients of all optimized
    optimizer.zero_grad()
    # back propagation algorithm
    loss.backward()
    # Performs a single optimization step (parameter update).
    optimizer.step()
    print("��%d��ѵ����lossΪ%.3f" % (i, loss.item()))
    loss_list.append(loss)
    x.append(i)

# Saves an object to a disk file.
torch.save(model.state_dict(),"../TrainedModel/LeNet5.pkl")
print('Networks''s keys: ', model.state_dict().keys())

plt.figure()
plt.xlabel("number of epochs")
plt.ylabel("loss")
plt.plot(x,loss_list,"r-")
plt.show()