import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


transform = transforms.Compose(
    [transforms.ToTensor(),
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
	def __init__(self):
		super(CapsuleNet,self).__init__()
		self.conv1 = nn.Conv2d(in_channels=3,out_channels=256,kernel_size=9,stride=1)
		self.primary_capsules()
		self.class_capsules()

	def forward(self,x,y=None):
		#x = [batch_size,3,32,32]
		#y = [batch_size,10]

		x = F.relu(self.conv1(x))
		x = self.primary_capsules(x)
		x = self.class_capsules(x)
		return x

	def loss()



