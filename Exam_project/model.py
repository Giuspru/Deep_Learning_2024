import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import models
from PIL import Image

'''
    Come si può vedere nel file nominato dimensioni.py abbiamo:
    resnet50: (3x256x144) -> (2048,8,5)
    resnet18: (3x256x144) -> (512,8,5)
    Partiamo da questo per creare il nostro decoder, che deve riportare le features maps ottenute
    ad una dimensione (1,256,144)

'''

#Class implementation: 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #prova = models.efficientnet_b0(weights="IMAGENET1K_V1", pretrained=True)
        resnet = models.resnet50(weights="IMAGENET1K_V1" )
        self.resnet_backbone = list(resnet.children())[:-2] 
        self.encoder = nn.Sequential(*self.resnet_backbone)

        for i, layer in enumerate(self.encoder):
            if i < len(self.encoder) - 2:
                for param in layer.parameters():
                    param.requires_grad = False
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=2, padding=0),  #(9x15)
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1), #Same size
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=0), #(19x31)
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1), #Same size
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=0), #(39x63)
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1), #Same size
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=0), #(79x127)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1), #Same size
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=0), #(159x255)     
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), #Same size
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=2, padding=0), #(319x511) 
            nn.AdaptiveMaxPool2d(output_size=(144, 256)) #Reshaping to (1,144,256)

            )

    
    
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
    
