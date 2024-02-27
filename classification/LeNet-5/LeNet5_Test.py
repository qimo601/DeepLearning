import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyTorchVersion.Networks.LeNet5 import LeNet5

model = LeNet5()
test_data = pd.DataFrame(pd.read_csv("../Data/mnist_test.csv"))
#Load model parameters
model.load_state_dict(torch.load("../TrainedModel/LeNet5.pkl"))

accuracy_list = []
testList = []

with torch.no_grad():
    # ����һ�ٴβ���
    for i in range(100):
        # ÿ�δӲ��Լ��������ѡ50������
        batch_data = test_data.sample(n=50,replace=False)
        batch_x = torch.from_numpy(batch_data.iloc[:,1::].values).float().view(-1,1,28,28)
        batch_y = batch_data.iloc[:,0].values
        prediction = np.argmax(model(batch_x).numpy(), axis=1)
        acccurcy = np.mean(prediction==batch_y)
        print("��%d����Լ���׼ȷ��Ϊ%.3f" % (i,acccurcy))
        accuracy_list.append(acccurcy)
        testList.append(i)

plt.figure()
plt.xlabel("number of tests")
plt.ylabel("accuracy rate")
plt.ylim(0,1)
plt.plot(testList, accuracy_list,"r-")
plt.legend()
plt.show()