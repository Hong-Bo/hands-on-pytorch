"""An implementation of SEResNet-18 to classify CIFAR-10 dataset

The model is slightly tailored in several ways. The original network
was designed for dataset with much higher resolution while images in CIFAR-10
only have resolution of 32 * 32. Due to resolution discrepancy, this version
of SEResNet-18 has smaller kernel sizes and stride paces in one way and the fully
connected layers are reduced to only one layer in another.

With being trained for 100 epochs, the accuracy of this network is around 92%.

Example:
    # Classifying a CIFAR-10 image using this module

    # Load the Classifier class from this module
    c = Classifier('../data', force_training=False)

    test_data = Data('../data', test_batch_size=1).test_loader
    test_data = iter(test_data)
    image, label = test_data.next()

    import time
    start = time.time()
    predict = c.predict(image.to(c.device))[0]
    end = time.time()
    logger.info("Prediction of the test image ({}): {}".format(label, predict))
    logger.info("Time consumed to predict: {}".format(end - start))

Reference:
    https://arxiv.org/pdf/1709.01507.pdf

TODO:

"""
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import OrderedDict

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Data(object):
    """Prepare CIFAR-10 data for training and testing.

    Args:
        data_dir (str): directory to save downloaded data
        train_batch_size (int): batch size of training data, defaulted 64
        test_batch_size (int): batch size of testing data, defaulted 256

    Attributes:
        data_dir (str): directory to save downloaded data
        transform (opr): operator to transform MNIST data
        train_loader (ite): an iterator of loading training data
        test_loader (ite): an iterator of loading testing data

    Example:
        1. Iterate data
        # Initializing
        data = Data(data_dir='../data')
        # Loading training data:
        for batch_idx, (images, labels) in enumerate(data.train_loader):
            print("image size = {}".format((images.size())))
            print("labels = {}\n{}".format(labels.size(), labels))

        2. Load data
        data = Data(data_dir='../data')
        train_data = data.dataset(train=True)
        image100, label100 = train_data[100][0], train_data[100][1]
        print(image100.size(), label100)
    """
    def __init__(self, data_dir, train_batch_size=64, test_batch_size=256):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.train_loader = self._loader(train=True, batch_size=train_batch_size)
        self.test_loader = self._loader(train=False, batch_size=test_batch_size)

    def _loader(self, train, batch_size):
        dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=train, transform=self.transform, download=True
        )

        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True
        )

    def dataset(self, train=True):
        return torchvision.datasets.MNIST(
            root=self.data_dir, train=train, transform=self.transform, download=True
        )


class ResidualBlock(nn.Module):
    """An implementation of Residual Module

    Example:
        inputs = torch.ones([2, 3, 32, 32])
        res_block = ResidualBlock(in_channels=3, out_channels=64, stride=1)
        outputs = res_block(inputs)
        print("Size of Output:", outputs.size())

        res_block = ResidualBlock(in_channels=3, out_channels=64, stride=2)
        outputs = res_block(inputs)
        print("With Down Sample:", outputs.size())
    """
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(ResidualBlock, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv3_0', self._conv3x3(in_channels, out_channels, stride)),
            ('bn_0', nn.BatchNorm2d(num_features=out_channels)),
            ('relu_0', nn.ReLU(inplace=True)),
            ('conv3_1', self._conv3x3(out_channels, out_channels, stride=1)),
            ('bn_1', nn.BatchNorm2d(num_features=out_channels)),
            ('relu_1', nn.ReLU(inplace=True)),
        ]))
        self.se_layers = nn.Sequential(OrderedDict([
            ('avg_pool', nn.AdaptiveAvgPool2d(output_size=1)),
            ('fc1', nn.Conv2d(out_channels, out_channels//reduction, kernel_size=1)),
            ('relu', nn.ReLU(inplace=True)),
            ('fc2', nn.Conv2d(out_channels//reduction, out_channels, kernel_size=1)),
            ('sigmoid', nn.Sigmoid())
        ]))
        self.down_sample = self._down_sample(in_channels, out_channels, stride)

    @staticmethod
    def _conv3x3(in_channels, out_channels, stride):
        """
        Construct a convolutional layer with kernel size = 3 * 3
        :param in_channels: integer, the input channels
        :param out_channels: integer, the output channels
        :param stride: integer, convolution stride
        :return: a convolutional layer
        """
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

    @staticmethod
    def _down_sample(in_channels, out_channels, stride=1):
        if stride != 1 or in_channels != out_channels:
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        return None

    def forward(self, x):
        out = self.features(x)
        w = self.se_layers(out)
        out = out * w
        if self.down_sample:
            x = self.down_sample(x)
        out += x
        return F.relu(out, inplace=True)


class SEResNet18(nn.Module):
    """An implementation of SEResNet-18

    Args:

    Attributes:
        features, sequential convolutional  layers
        fc, sequential fully connected layers

    Example:
        model = SEResNet18(10)
        inputs = torch.ones([3, 3, 32, 32])
        outputs = model(inputs)
        print(outputs)
    """
    def __init__(self, num_classes=10):
        super(SEResNet18, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )
        self.res_layers = self._make_layer([2]*4, [64, 128, 256, 512], [1, 2, 2, 2], 64)

        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(layers, out_channels, strides, in_channel):
        residual_layers = []
        for para in zip(layers, out_channels, strides):
            residual_layers.append(ResidualBlock(in_channel, para[1], para[2]))
            for i in range(1, para[0]):
                residual_layers.append(ResidualBlock(para[1], para[1]))
            in_channel = para[1]
        return nn.Sequential(*residual_layers)

    def forward(self, images):
        out = self.pre_layers(images)
        out = self.res_layers(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Classifier(object):
    """To classify a CIFAR-10 image
    Args:
        data_dir (path), directory of CIFAR-10 data
        model_dir (path), directory to load and save model files
        log_interval (int), interval of logging when training model
        epochs (int), how many epochs to train the model
        lr (float), learning rate of training
        momentum (float), momentum of learning
        force_training (bool), whether to train the model even it has been trained before

    Example:
        # Predict the label of a random image
        c = Classifier('../data', force_training=False)

        test_data = Data('../data', test_batch_size=1).test_loader
        test_data = iter(test_data)
        image, label = test_data.next()

        import time
        start = time.time()
        predict = c.predict(image.to(c.device))[0]
        end = time.time()
        logger.info("Prediction of the test image ({}): {}".format(label, predict))
        logger.info("Time consumed to predict: {}".format(end - start))
    """
    def __init__(self, data_dir, model_dir='../data/seresnet18.pth', log_interval=50,
                 epochs=100, lr=0.01, momentum=0.5, force_training=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SEResNet18(10).to(self.device)
        self.data = Data(data_dir)
        self.epochs = epochs
        self.log_interval = log_interval
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.model_loaded = self.load_model(model_dir, force_training)

    def load_model(self, model_dir, force_training):
        if os.path.exists(model_dir) and not force_training:
            self.model.load_state_dict(torch.load(model_dir, map_location='cpu'))
        else:
            logger.info("No pre-trained model is found. Start training from scratch")
            for epoch in range(1, self.epochs + 1):
                self.train(epoch)
                self.test()
            torch.save(self.model.state_dict(), model_dir)
        self.model.eval()
        return True

    def train(self, epoch):
        self.model.train()
        for batch_idx, (images, labels) in enumerate(self.data.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            self.optimizer.step()
            if (batch_idx+1) % self.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch, (batch_idx + 1) * len(images), len(self.data.train_loader.dataset),
                        100. * batch_idx / len(self.data.train_loader), loss.item()
                    )
                )

    def test(self):
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for images, labels in self.data.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                output = self.model(images)
                test_loss += F.cross_entropy(output, labels, reduction='sum').item()

                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()

        len_data = len(self.data.test_loader.dataset)
        test_loss /= len_data
        logger.info("\nTest results: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len_data, 100. * correct/len_data
        ))

    def predict(self, img):
        output = self.model(img)
        _, predicted = output.max(1)
        return predicted


if __name__ == "__main__":
    c = Classifier('../data', force_training=False)

    test_data = Data('../data', test_batch_size=1).test_loader
    test_data = iter(test_data)
    image, label = test_data.next()

    import time
    start = time.time()
    predict = c.predict(image.to(c.device))[0]
    end = time.time()
    logger.info("Prediction of the test image ({}): {}".format(label, predict))
    logger.info("Time consumed to predict: {}".format(end - start))
