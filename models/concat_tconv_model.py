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
            nn.BatchNorm2d(3, 1e-05, 0.1, True, track_running_stats = True),
            nn.ReLU()
        )
        # 8
        self.d0 = nn.Sequential(
            nn.Conv2d(3, 4, 3, 1, 1), 
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.BatchNorm2d(4, 1e-05, 0.1, True, track_running_stats = True),
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )
        # 4
        self.d1 = nn.Sequential(
            nn.Conv2d(4, 6, 3, 1, 1),
            nn.Conv2d(6, 6, 3, 1, 1),
            nn.BatchNorm2d(6, 1e-05, 0.1, True, track_running_stats = True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 2
        self.d2 = nn.Sequential(
            nn.Conv2d(6, 8, 3, 1, 1),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.Dropout(0.6),
            nn.MaxPool2d(2)
        )
        # 2
        self.connect = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8, 1e-05, 0.1, True, track_running_stats = True),
            nn.ReLU(),
        )
        #cat(d2, connect) to up0
        # 4 
        #cat(up0, d1) to up1
        self.up0 = nn.Sequential(
            nn.Conv2d(16, 6, 3, 1, 1),
            nn.Conv2d(6, 6, 3, 1, 1),
            nn.Dropout(0.6),
            nn.BatchNorm2d(6, 1e-05, 0.1, True, track_running_stats = True),
            nn.ConvTranspose2d(6, 6, 4, 2, 1),
            nn.ReLU(),
        )
        # 8
        self.up1 = nn.Sequential(
            nn.Conv2d(12, 4, 3, 1, 1),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.BatchNorm2d(4, 1e-05, 0.1, True, track_running_stats = True),
            nn.ConvTranspose2d(4, 4, 4, 2, 1),
            nn.ReLU(),
            
        )
        #16
        #cat(up1, d0) to up2
        self.up2 = nn.Sequential(
            nn.Conv2d(4, 4, 3, 1, 1),    
            nn.Conv2d(4, 1, 3, 1, 1),
            nn.BatchNorm2d(1, 1e-05, 0.1, True, track_running_stats = True),
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
summary(model, (1, 240, 416))

