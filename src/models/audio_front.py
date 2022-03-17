from torch import nn
from src.models.resnet import BottleNeck1D_IR, BottleNeck_IR, Flatten
import torch

class Audio_front(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.in_channels = in_channels

        self.frontend = nn.Sequential(
            nn.Conv2d(self.in_channels, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.PReLU(256)
        )

        self.Res_block = nn.Sequential(
            BottleNeck_IR(256, 256, stride=1, dim_match=True),
            nn.BatchNorm2d(256),
            nn.Dropout(0.4),
        )

        self.Linear = nn.Sequential(nn.Linear(256 * 20, 512),
                                    nn.LayerNorm(512))

    def forward(self, x):
        x = self.frontend(x)    #B, 512, F/4, T/4
        x = self.Res_block(x)  #B, 512, F/4, T/4
        b, c, f, t = x.size()
        x = x.view(b, c*f, t).transpose(1, 2).contiguous() #B, T/4, 512 * F/4
        x = x.view(b*t, -1)
        x = self.Linear(x)      # B, T/4, 512             #0.23 sec (23 frames)
        x = x.view(b, t, -1)
        return x

