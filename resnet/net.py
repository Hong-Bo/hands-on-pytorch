import torch.nn as nn
import torchvision.transforms as transforms


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.downsample = self.make_downsample(in_channels, out_channels, stride)
        self.conv1 = self.make_relu_block(in_channels, out_channels, stride=stride)
        # Notice that the stride for the 2ed convolutional layer is defaulted to be 1!
        self.conv2 = self.make_block(out_channels, out_channels)

    @staticmethod
    def make_block(in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    @staticmethod
    def make_relu_block(in_channels, out_channels, stride=1):
        return nn.Sequential(
            ResBlock.make_block(in_channels, out_channels, stride),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def make_downsample(in_channels, out_channels, stride):
        if (in_channels != out_channels) or (stride != 1):
            return ResBlock.make_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        else:
            return None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            residual = self.downsample(x)
        out = residual + out
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = ResBlock.make_relu_block(in_channels=3, out_channels=16, stride=1)
        self.layer1 = self.make_layer(block, out_channels=16, blocks=layers[0])
        self.layer2 = self.make_layer(block, out_channels=32, blocks=layers[1], stride=2)
        self.layer3 = self.make_layer(block, out_channels=64, blocks=layers[2], stride=2)
        self.layer4 = self.make_layer(block, out_channels=64, blocks=layers[3], stride=2)
        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(self.in_channels, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = list()
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)   # [10, 16, 32, 32]
        out = self.layer1(out)  # [10, 16, 32, 32]
        out = self.layer2(out)  # [10, 32, 16, 16]
        out = self.layer3(out)  # [10, 64, 8, 8]
        out = self.layer4(out)  # [10, 64, 4, 4]
        out = self.avg_pool(out)  # 10, 64, 1, 1]
        out = out.view(out.size(0), -1)  # [10, 64]
        out = self.fc(out)  # [10, 10]
        return out


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()
    ])

    from data import Data
    data = Data(data_dir='../data', batch_size=10, transform=transform)
    net_block = ResBlock(in_channels=3, out_channels=16, stride=2)
    net = ResNet(ResBlock, layers=[2, 2, 2, 2])

    for i, (images, labels) in enumerate(data.train_loader):
        print("size of images = {}".format((images.size())))
        print("size of labels = {}".format((labels.size())))
        output = net_block(images)
        print("size of output = {}".format(output.size()))
        output = net(images)
        print("size of output = {}".format(output.size()))
        break
