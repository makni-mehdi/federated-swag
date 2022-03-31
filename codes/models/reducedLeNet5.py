import torch.nn as nn
import torchvision.transforms as transforms
import torch

__all__=['reducedLeNet5']

class ModelBase(nn.Module):
    def __init__(self, output_dim=10):
        super(ModelBase, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, output_dim)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        return y
    
    
class reducedLeNet5:
    base = ModelBase
    args = list()
    kwargs = {'output_dim' : 10}
    transform_train = transforms.ToTensor()
    transform_test = transforms.ToTensor()

