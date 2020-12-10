import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 包含一个卷积层和池化层，分别对应LeNet5中的C1和S2，
        # 卷积层的输入通道为1，输出通道为6，设置卷积核大小5x5，步长为1
        # 池化层的kernel大小为2x2
        self._conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        # 包含一个卷积层和池化层，分别对应LeNet5中的C3和S4，
        # 卷积层的输入通道为6，输出通道为16，设置卷积核大小5x5，步长为1
        # 池化层的kernel大小为2x2
        self._conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        # 对应LeNet5中C5卷积层，由于它跟全连接层类似，所以这里使用了nn.Linear模块
        # 卷积层的输入通特征为4x4x16，输出特征为120x1
        self._fc1 = nn.Sequential(
            nn.Linear(in_features=4*4*16, out_features=120)
        )
        # 对应LeNet5中的F6，输入是120维向量，输出是84维向量
        self._fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84)
        )
        # 对应LeNet5中的输出层，输入是84维向量，输出是10维向量
        self._fc3 = nn.Sequential(
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, input):
        # 前向传播
        # MNIST DataSet image's format is 28x28x1
        # [28,28,1]--->[24,24,6]--->[12,12,6]
        conv1_output = self._conv1(input)
        # [12,12,6]--->[8,8,,16]--->[4,4,16]
        conv2_output = self._conv2(conv1_output)
        # 将[n,4,4,16]维度转化为[n,4*4*16]
        conv2_output = conv2_output.view(-1, 4 * 4 * 16)
        # [n,256]--->[n,120]
        fc1_output = self._fc1(conv2_output)
        # [n,120]-->[n,84]
        fc2_output = self._fc2(fc1_output)
        # [n,84]-->[n,10]
        fc3_output = self._fc3(fc2_output)
        return fc3_output



