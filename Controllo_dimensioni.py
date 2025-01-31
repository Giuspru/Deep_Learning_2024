import torch 
from torchvision import models
import torch.nn as nn
from model  import Net

resnet = models.resnet50(weights='IMAGENET1K_V1')
backbone = nn.Sequential(*list(resnet.children())[:-2])

x = torch.randn(1, 3, 144, 256)
y = backbone(x)

print(y.shape)  

resnet2 = models.resnet18(weights='IMAGENET1K_V1')
backbonekbone2 = nn.Sequential(*list(resnet2.children())[:-2])

y2 = backbonekbone2(x)
print(y2.shape)

net = Net()
pred = net(x)
print(x.shape)
print(pred.shape)