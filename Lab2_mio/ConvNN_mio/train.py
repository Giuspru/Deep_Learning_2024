import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from ConvNN_mio.deeper_net import DeeperNet

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


def evaluate(net, loader, device):
    net.eval()
    num_correct, num_total = 0, 0

    
    with torch.no_grad():
        for inputs in loader:
            images = inputs[0].to(device)
            labels = inputs[1].to(device)

            outputs = net(images)
            _, preds = torch.max(outputs.detach(), 1)

            num_correct += (preds == labels).sum().item()
            num_total += labels.size(0)

    return num_correct / num_total


def train(args):
    # prepare the MNIST dataset
    train_dataset = datasets.MNIST(root='./data',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)
    
    test_dataset = datasets.MNIST(root='./data',
                                  train=False,
                                  transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size,
                                              shuffle=False)
    
    # define the device:
    # turn on the CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = DeeperNet().to(device)

    #loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(args.max_epochs):
        net.train()
        for step, (images, labels) in enumerate(train_loader):

            # forward-propagation: Le immagini sono già in un formato adeguato.
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = loss_fn(outputs, labels)

            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = evaluate(net, test_loader, device)
        print("Epoch [{}/{}] loss: {:.5f} test acc: {:.3f}"
              .format(epoch + 1, args.max_epochs, loss.item(), acc))

    torch.save(net.state_dict(), "mnist-final.pth")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()