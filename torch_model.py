import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch
from metoloss import m3loss

#wo = (wi - F + 2P)/S + 1


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        # in, out, kernel, stride, pad -> conv2d
        # kernel, stride -> max pool
        # nn.BatchNorm2d(8, 1e-05, 0.1, True, track_running_stats = True),
        # nn.ConvTranspose2d(1, 1, 4, 2, 0)

        # 16 
        self.input = nn.Sequential(
            nn.Conv2d(1, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace = True)
        )
        # 8
        self.d0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), 
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2) 
        )
        # 4
        self.d1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2)
        )
        # 2
        self.d2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.Dropout(0.6),
            nn.MaxPool2d(2)
        )
        # 2
        self.connect = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
        )
        #cat(d2, connect) to up0
        # 4 
        #cat(up0, d1) to up1
        self.up0 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.Dropout(0.6),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.ReLU(inplace = True),
        )
        # 8
        self.up1 = nn.Sequential(
            nn.Conv2d(320, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.ReLU(inplace = True),
            
        )
        #16
        #cat(up1, d0) to up2
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, 5, 1, 2),    
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 32, 5, 1, 2),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(16, 8, 5, 1, 2),
            nn.ReLU(inplace = True),
            nn.Conv2d(8, 1, 3, 1, 1),
            nn.ConvTranspose2d(1, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        first = self.input(x)
        d0 = self.d0(first)
        d1 = self.d1(d0)
        d2 = self.d2(d1)
        con = self.connect(d2)
        u0 = self.up0(torch.cat((d2, con), 1))
        u1 = self.up1(torch.cat((d1, u0), 1))
        u2 = self.up2(u1)
        return u2



def w_b_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

model = network().cuda()
print('$ pip3 install torchsummary')
print('from torchsummary import summary')
print('>>>model = network().cuda()')
print('>>>summary(model, input_size = (1, 240, 416), batch_size = 4)')
summary(model, input_size = (1, 240, 416), batch_size = 4)   

