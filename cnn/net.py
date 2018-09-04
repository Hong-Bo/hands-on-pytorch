import torch.nn as nn
import torchvision.transforms as transforms
from .data import data


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, *input):
        out = self.layer1(input)
        return out


if __name__ == "__main__":
    data = data.Data(data_dir='../data', batch_size=100, transform=transforms.ToTensor())
    model = ConvNet()
    element = data.train_dataset[0][0]
    output = model(element)
    print(output)
