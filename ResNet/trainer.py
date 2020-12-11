import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torch.nn as nn

class Trainer(object):
    def __init__(self, model, config):
        self._model = model
        self._config = config
        self._optimizer = torch.optim.Adam(self._model.parameters(),\
                                           lr=config['lr'], weight_decay=config['l2_regularization'])
        self.loss_func = nn.CrossEntropyLoss()

    def _train_single_batch(self, images, labels):
        """
        对单个小批量数据进行训练
        """
        y_predict = self._model(images)

        loss = self.loss_func(y_predict, labels)
        # 先将梯度清零,如果不清零，那么这个梯度就和上一个mini-batch有关
        self._optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 梯度下降等优化器 更新参数
        self._optimizer.step()
        # 将loss的值提取成python的float类型
        loss = loss.item()

        # 计算训练精确度
        # 这里的y_predict是一个多个分类输出，将dim指定为1，即返回每一个分类输出最大的值以及下标
        _, predicted = torch.max(y_predict.data, dim=1)
        return loss, predicted

    def _train_an_epoch(self, train_loader, epoch_id):
        """
        训练一个Epoch，即将训练集中的所有样本全部都过一遍
        """
        # 设置模型为训练模式，启用dropout以及batch normalization
        self._model.train()

        total = 0
        correct = 0

        # 从DataLoader中获取小批量的id以及数据
        for batch_id, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)
            if self._config['use_cuda'] is True:
                images, labels = images.cuda(), labels.cuda()

            loss, predicted = self._train_single_batch(images, labels)

            # 计算训练精确度
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

            print('[Training Epoch: {}] Batch: {}, Loss: {}'.format(epoch_id, batch_id, loss))
        print('Training Epoch: {}, accuracy rate: {}%%'.format(epoch_id, correct / total * 100.0))

    def train(self, train_dataset):
        # 是否使用GPU加速
        self.use_cuda()
        for epoch in range(self._config['num_epoch']):
            print('-' * 20 + ' Epoch {} starts '.format(epoch) + '-' * 20)
            # 构造DataLoader
            data_loader = DataLoader(dataset=train_dataset, batch_size=self._config['batch_size'], shuffle=True)
            # 训练一个轮次
            self._train_an_epoch(data_loader, epoch_id=epoch)

    def use_cuda(self):
        if self._config['use_cuda'] is True:
            assert torch.cuda.is_available(), 'CUDA is not available'
            torch.cuda.set_device(self._config['device_id'])
            self._model.cuda()

    def save(self):
        self._model.saveModel()
