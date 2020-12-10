import torch
from AlexNet.network import AlexNet
from AlexNet.trainer import Trainer
from AlexNet.dataloader import LoadCIFAR10
from AlexNet.dataloader import Construct_DataLoader
from torch.autograd import Variable

alexnet_config = \
{
    'num_epoch': 5,
    'batch_size': 500,
    'lr': 1e-3,
    'l2_regularization':1e-4,
    'num_classes': 10,
    'device_id': 0,
    'use_cuda': False,
    'model_name': '../TrainedModels/AlexNet.model'
}

if __name__ == "__main__":
    ####################################################################################
    # AlexNet 模型
    ####################################################################################
    train_dataset, test_dataset = LoadCIFAR10(True)
    # define AlexNet model
    alexNet = AlexNet(alexnet_config)

    ####################################################################################
    # 模型训练阶段
    ####################################################################################
    # # 实例化模型训练器
    # trainer = Trainer(model=alexNet, config=alexnet_config)
    # # 训练
    # trainer.train(train_dataset)
    # # 保存模型
    # trainer.save()

    ####################################################################################
    # 模型测试阶段
    ####################################################################################
    alexNet.eval()
    alexNet.loadModel(map_location=torch.device('cpu'))
    if alexnet_config['use_cuda']:
        alexNet = alexNet.cuda()

    correct = 0
    total = 0
    for images, labels in Construct_DataLoader(test_dataset, alexnet_config['batch_size']):
        images = Variable(images)
        labels = Variable(labels)
        if alexnet_config['use_cuda']:
            images = images.cuda()
            labels = labels.cuda()

        y_pred = alexNet(images)
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        temp = (predicted == labels.data).sum()
        correct += temp
    print('Accuracy of the model on the test images: %.2f%%' % (100.0 * correct / total))



