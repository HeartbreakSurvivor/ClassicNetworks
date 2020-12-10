import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import matplotlib.pyplot as plt
from LeNet5.LeNet5 import LeNet5

train_data = pd.DataFrame(pd.read_csv("../Data/MNIST/mnist_train.csv"))

model = LeNet5()
print(model)

# 定义交叉熵损失函数
loss_fc = nn.CrossEntropyLoss()
# 用model的参数初始化一个随机梯度下降优化器
optimizer = optim.SGD(params=model.parameters(),lr=0.001, momentum=0.78)
loss_list = []
x = []

# 迭代次数1000次
for i in range(1000):
    # 小批量数据集大小设置为30
    batch_data = train_data.sample(n=30, replace=False)
    # 每一条数据的第一个值是标签数据
    batch_y = torch.from_numpy(batch_data.iloc[:,0].values).long()
    #图片信息，一条数据784维将其转化为通道数为1，大小28*28的图片。
    batch_x = torch.from_numpy(batch_data.iloc[:,1::].values).float().view(-1,1,28,28)

    # 前向传播计算输出结果
    prediction = model.forward(batch_x)
    # 计算损失值
    loss = loss_fc(prediction, batch_y)
    # Clears the gradients of all optimized
    optimizer.zero_grad()
    # back propagation algorithm
    loss.backward()
    # Performs a single optimization step (parameter update).
    optimizer.step()
    print("第%d次训练，loss为%.3f" % (i, loss.item()))
    loss_list.append(loss)
    x.append(i)

# Saves an object to a disk file.
torch.save(model.state_dict(),"../TrainedModels/LeNet5.pkl")
print('Networks''s keys: ', model.state_dict().keys())

plt.figure()
plt.xlabel("number of epochs")
plt.ylabel("loss")
plt.plot(x,loss_list,"r-")
plt.show()
