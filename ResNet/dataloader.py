import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def Construct_DataLoader(dataset, batchsize):
    return DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True)

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def LoadCIFAR10(download=False):
    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='../Data/CIFAR10', train=True, transform=transform, download=download)
    test_dataset = torchvision.datasets.CIFAR10(root='../Data/CIFAR10', train=False, transform=transform)
    return train_dataset, test_dataset
