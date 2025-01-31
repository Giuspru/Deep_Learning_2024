import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import models
from PIL import Image

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, stride=1),
            nn.AdaptiveMaxPool2d(output_size=(144, 256))
        )

    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out