from pipeline import Pipeline
from net import ResNet, ResBlock
from data import Data
import torchvision.transforms as transforms


def main():
    resnet = ResNet(ResBlock, layers=[2, 2, 2])

    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()
    ])
    cifar10 = Data(data_dir='../data', batch_size=100, transform=transform)

    pipe = Pipeline(resnet, cifar10, lr=0.1, momentum=0.9, log_interval=100, epochs=40)
    pipe.run()


if __name__ == '__main__':
    main()
