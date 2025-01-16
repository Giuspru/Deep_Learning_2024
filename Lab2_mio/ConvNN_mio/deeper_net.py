import torch 
import torch.nn as nn



class DeeperNet(nn.Module):
    def __init__(self):
        super(DeeperNet, self).__init__()

        self.cov1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 , stride=2)
        ) #Dimensione 14x14

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1 , stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 , stride=2)
        ) #Dimensione 7x7

        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1 , padding=0)
            ) #Dimensione 7x7

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1 , padding=0)
        ) #Dimensione 7x7

        self.fc = nn.Linear(7*7*256,10)

        

    def forward(self, X):
        conv1 = self.cov1(X)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        flat = conv4.view(conv4.size(0), -1)
        out = self.fc(flat)
        return out
    
#print(DeeperNet())
#print(DeeperNet().parameters())

        