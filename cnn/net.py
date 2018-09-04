import torch.nn as nn
import torchvision.transforms as transforms
from data import Data


class ConvNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)  # out.size(0) is the batch size
        out = self.fc(out)
        return out


if __name__ == "__main__":
    data = Data(data_dir='../data', batch_size=100, transform=transforms.ToTensor())
    model = ConvNet(input_size=7*7*32, output_size=10)
    for i, (images, labels) in enumerate(data.train_loader):
        print("size of images = {}".format(images.size()))
        output = model(images)
        print("size of output = {}".format(output.size()))
        break

    # print(type(data.train_dataset[0]))
    # element = data.train_dataset[0][0]
    # print(element.size())
    # print(type(element))
    # output = model(element)
    # print(output)
