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

        self.d0 = nn.Sequential(
            nn.Conv2d(1, 8, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(8, 1e-05, 0.1, True, track_running_stats = True),
            nn.Conv2d(8, 8, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(8, 1e-05, 0.1, True, track_running_stats = True),
            nn.MaxPool2d(2)   
        )

        self.d1 = nn.Sequential(
            nn.Conv2d(8, 16, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(16, 1e-05, 0.1, True, track_running_stats = True),
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(16, 1e-05, 0.1, True, track_running_stats = True),
            nn.MaxPool2d(2) 
        )

        self.d2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.BatchNorm2d(32, 1e-05, 0.1, True, track_running_stats = True),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(32, 1e-05, 0.1, True, track_running_stats = True),
            nn.MaxPool2d(2) 
        )

        self.u0 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(16, 1e-05, 0.1, True, track_running_stats = True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(16, 1e-05, 0.1, True, track_running_stats = True),
            nn.UpsamplingBilinear2d(scale_factor = 2)
        )

        self.u1 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(8, 1e-05, 0.1, True, track_running_stats = True),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(8, 1e-05, 0.1, True, track_running_stats = True),
            nn.UpsamplingBilinear2d(scale_factor = 2)
        )

        self.u2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(8, 1e-05, 0.1, True, track_running_stats = True),
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(4, 1e-05, 0.1, True, track_running_stats = True),
            nn.Conv2d(4, 1, 3, 1, 1),
            nn.UpsamplingBilinear2d(scale_factor = 2),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.d0(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.u0(x)
        x = self.u1(x)
        x = self.u2(x)
        return x



def w_b_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)


