'''
Patrick Chao and Brenton Chu
2/28/18
'''

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from capsule_layer import CapsuleLayer



M_PLUS = 0.9
M_MINUS = 0.1
LAMBDA = 0.5


class CapsuleNet(nn.Module):
    def __init__(self, use_cuda=False, image_size=[3, 32, 32], unit_size=16, num_classes=10,
                 fc1_size=512, fc2_size=1024):
        super(CapsuleNet,self).__init__()
        self.use_reconstruction_loss = True
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=256,kernel_size=9,stride=1)
        self.primary = CapsuleLayer(in_capsules=-1, in_channels=256, out_capsules=8,
                                    unit_size=-1, use_routing=False, num_iters=0,
                                    use_cuda=use_cuda)
        pch = self.primary.size(1)
        self.classes = CapsuleLayer(in_capsules=8, in_channels=pch, out_capsules=num_classes,
                                    unit_size=unit_size, use_routing=True, num_iters=3,
                                    use_cuda=use_cuda)

        self.decoder = n.Sequential(
        nn.Linear(unit_size * num_classes, fc1_size),
        nn.ReLU(inplace=True),
        nn.Linear(fc1_size, fc2_size),
        nn.ReLU(inplace=True),
        nn.Linear(fc2_size, reduce(lambda x, y: x * y, image_size, 1))
        nn.Sigmoid()
        )

    def forward(self,x,y=None):
        #x = [batch_size,3,32,32]
        #y = [batch_size,10]

        x = F.relu(self.conv1(x))
        x = self.primary(x)
        x = self.classes(x)

        # Do reconstruction
        if y = None:
          # Get max index
           _, max_length_indices = torch.norm(x,p=2,dim=3).squeeze(2).max(dim=1)
          y = Variable(torch.sparse.torch.eye(x.size(3))).index_select(dim=0, index=max_length_indices.data)
        
        mask = y.unsqueeze(2).unsqueeze(3)
        masked = mask*x

        r = x.view(x.size(0), -1)
        r = F.relu(self.r_fc1(r))
        r = F.relu(self.r_fc2(r))
        r = F.sigmoid(self.r_fc3(r))

        return x

    def loss(self,image, class_capsules, labels):
        batch_size = image.size(0)
        recon_loss = 0
        m_loss = self.margin_loss(class_capsules, labels).mean()

        total_loss = m_loss
        if self.use_reconstruction_loss:
          reconstructed = self.decoder(image)
          recon_loss = self.reconstruction_loss(reconstruction,image)

        return (m_loss + 0.0005 * recon_loss) / batch_size


    def margin_loss(self,class_capsules,labels):
      # input batchsize, out_capsules,1,unit size
      batch_size = class_capsules.size(0)
      normed_capsules = torch.norm(class_capsules,p=2,dim=3) #batchsize,out_capsules,1
      zero = Variable(torch.zeros(1),requires_grad=False)
      left_max = labels*torch.max(zero,M_PLUS-normed_capsules).view(batch_size,-1)**2
      right_max = LAMBDA*(1.0-labels)*torch.max(zero,normed_capsules-M_MINUS).view(batch_size,-1)**2
      loss =  left_max + right_max
      total_loss = loss.sum(dim=1)
      return total_loss


    def reconstruction_loss(self, reconstructed, image):
      image = image.view(image.shape[0], -1)
      error = torch.sum((reconsructed - image)**2)
      return error