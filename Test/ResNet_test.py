import torch
from ResNet.network import ResNet
from ResNet.network import BasicBlock
from ResNet.network import Bottleneck
from ResNet.trainer import Trainer
from ResNet.dataloader import LoadCIFAR10
from ResNet.dataloader import Construct_DataLoader
from torch.autograd import Variable

resnet_config = \
{
    'block_type': BasicBlock,
    'num_blocks': [2,2,2,2], #ResNet18
    'num_epoch': 20,
    'batch_size': 500,
    'lr': 1e-3,
    'l2_regularization':1e-4,
    'num_classes': 10,
    'device_id': 0,
    'use_cuda': True,
    'model_name': '../TrainedModels/ResNet18.model'
}

if __name__ == "__main__":
    ####################################################################################
    # AlexNet 模型
    ####################################################################################
    train_dataset, test_dataset = LoadCIFAR10(True)
    # define AlexNet model
    resNet = ResNet(resnet_config)

    ####################################################################################
    # 模型训练阶段
    ####################################################################################
    # 实例化模型训练器
    trainer = Trainer(model=resNet, config=resnet_config)
    # 训练
    trainer.train(train_dataset)
    # 保存模型
    trainer.save()

    ####################################################################################
    # 模型测试阶段
    ####################################################################################
    resNet.eval()
    if resnet_config['use_cuda']:
        resNet.loadModel(map_location=torch.device('cpu'))
        resNet = resNet.cuda()
    else:
        resNet.loadModel(map_location=lambda storage, loc: storage.cuda(resnet_config['device_id']))

    correct = 0
    total = 0
    for images, labels in Construct_DataLoader(test_dataset, resnet_config['batch_size']):
        images = Variable(images)
        labels = Variable(labels)
        if resnet_config['use_cuda']:
            images = images.cuda()
            labels = labels.cuda()

        y_pred = resNet(images)
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        temp = (predicted == labels.data).sum()
        correct += temp
    print('Accuracy of the model on the test images: %.2f%%' % (100.0 * correct / total))



