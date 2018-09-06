import torch.nn as nn
import torchvision.transforms as transforms


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = self.make_block(in_channels, out_channels, stride=1)

    def make_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)

        if self.downsample:
            residual = self.downsample(x)
        out = residual + out
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
    model = ResBlock(3, 3)

    for i, (images, labels) in enumerate(data.train_loader):
        print("size of images = {}".format((images.size())))
        print("size of labels = {}".format((labels.size())))
        output = model(images)
        print("size of output = {}".format(output.size()))
        break
