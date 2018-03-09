import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from capsule_layer import CapsuleLayer

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class CapsuleNet(nn.Module):
    def __init__(self, use_cuda=False, image_size=[3, 32, 32], unit_size=16, num_classes=10,
                 fc1_size=512, fc2_size=1024):
        super(CapsuleNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=256,kernel_size=9,stride=1)
        self.primary = CapsuleLayer(in_capsules=-1, in_channels=256, out_capsules=8,
                                    unit_size=-1, use_routing=False, num_iters=0,
                                    use_cuda=use_cuda)
        pch = self.primary.size(1)
        self.classes = CapsuleLayer(in_capsules=8, in_channels=pch, out_capsules=num_classes,
                                    unit_size=unit_size, use_routing=True, num_iters=3,
                                    use_cuda=use_cuda)
        self.r_fc1 = nn.Linear(unit_size * num_classes, fc1_size)
        self.r_fc2 = nn.Linear(fc1_size, fc2_size)
        self.r_fc3 = nn.Linear(fc2_size, reduce(lambda x, y: x * y, image_size, 1))

    def forward(self,x,y=None):
        #x = [batch_size,3,32,32]
        #y = [batch_size,10]

        x = F.relu(self.conv1(x))
        x = self.primary_capsules(x)
        x = self.class_capsules(x)

        r = x.view(x.size(0), -1)
        r = F.relu(self.r_fc1(r))
        r = F.relu(self.r_fc2(r))
        r = F.sigmoid(self.r_fc3(r))

        return x

    def loss()



