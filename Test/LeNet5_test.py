import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LeNet5.LeNet5 import LeNet5

model = LeNet5()
test_data = pd.DataFrame(pd.read_csv("../Data/MNIST/mnist_test.csv"))
#Load model parameters
model.load_state_dict(torch.load("../TrainedModels/LeNet5.pkl"))

accuracy_list = []
testList = []

with torch.no_grad():
    # 进行一百次测试
    for i in range(100):
        # 每次从测试集中随机挑选50个样本
        batch_data = test_data.sample(n=50,replace=False)
        batch_x = torch.from_numpy(batch_data.iloc[:,1::].values).float().view(-1,1,28,28)
        batch_y = batch_data.iloc[:,0].values
        prediction = np.argmax(model(batch_x).numpy(), axis=1)
        acccurcy = np.mean(prediction==batch_y)
        print("第%d组测试集，准确率为%.3f" % (i,acccurcy))
        accuracy_list.append(acccurcy)
        testList.append(i)

print(np.mean(accuracy_list))
plt.figure()
plt.xlabel("number of tests")
plt.ylabel("accuracy rate")
plt.ylim(0,1)
plt.plot(testList, accuracy_list,"r-")
plt.legend()
plt.show()
