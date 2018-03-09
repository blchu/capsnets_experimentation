import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLayer(nn.Module):
    def __init__(self, in_capsules, in_channels, out_capsules, unit_size, use_routing, num_iters,
                 use_cuda, filter_size=9, stride=2):
        super(CapsuleLayer, self).__init__()
        self.in_capsules = in_capsules
        self.in_channels = in_channels
        self.out_capsules = out_capsules
        self.unit_size = unit_size
        self.use_routing = use_routing
        self.num_iters = num_iters
        self.use_cuda = use_cuda
        self.filter_size = filter_size
        self.stride = stride

        if self.use_routing:
            self.w = nn.Parameter(torch.randn(self.in_channels, self.out_capsules,
                                              self.in_capsules, self.unit_size))
        else:
            conv_list = [nn.Conv2d(self.in_channels, self.unit_size, self.filter_size, stride)
                         for unit in range(self.out_capsules)]
            self.conv_capsules = nn.ModuleList(conv_list)

    def forward(self, x):
        if self.use_routing:
            # x dims are [batch_size, in_channels, in_capsules]
            x = torch.stack([x] * self.out_capsules, dim=2)
            x = x.unsqueeze(3)
            w = torch.stack([self.w] * x.size(0), dim=0)
            # u_hat dims are [batch_size, in_channels, out_capsules, 1, unit_size]
            u_hat = x @ w
            b = torch.zeros((self.in_channels, self.out_capsules, 1))
            b = b.cuda() if self.use_cuda else b

            # do routing iterations
            for _ in range(self.num_iters):
                # c dims are [batch_size, in_channels, out_capsules, 1, 1]
                c = torch.stack([F.softmax(b, dim=2)] * x.size(0), dim=0)
                c = c.unsqueeze(3)
                s = (c * u_hat).sum(dim=1, keepdim=True)
                # v dims are [batch_size, 1, out_capsules, 1, unit_size]
                v = squash(s, 4)
                update = u_hat @ torch.cat([v] * self.in_channels, dim=1).transpose(3, 4)
                b += update.mean(dim=0, keepdim=False)

            return v.squeeze(1)

        else:
            # x dims are [batch_size, in_channels, width, height]
            capsules = [conv(x) for _,conv in enumerate(self.conv_capsules)]
            capsules = torch.stack(capsules, dim=1)
            capsules = capsules.view(x.size(0), self.out_capsules, -1)
            capsules = capsules.transpose(1, 2)
            return self.squash(capsules, 1)

    def squash(vec, dim):
        norm = torch.norm(vec, dim, keepdim=True)
        return vec*norm / (1 + norm**2)

