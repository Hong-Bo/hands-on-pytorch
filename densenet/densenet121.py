"""An implementation of DenseNet to classify CIFAR-10 dataset

The model is slightly tailored in several ways. The original network
was designed for dataset with much higher resolution while images in CIFAR-10
only have resolution of 32 * 32. Due to resolution discrepancy, this version
of DenseNet has smaller kernel sizes and stride paces in one way and the fully
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
    https://arxiv.org/pdf/1608.06993.pdf

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


class DenseLayer(nn.Module):
    """An implementation of Inception module

    Example:
        inputs = torch.ones([2, 64, 32, 32])
        dense = DenseLayer(in_channels=64, bn_size=4, growth_rate=32)
        outputs = dense(inputs)
        print("Size of Output:", outputs.size())
    """
    def __init__(self, in_channels, bn_size, growth_rate):
        super(DenseLayer, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=bn_size*growth_rate,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=bn_size*growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bn_size*growth_rate, out_channels=growth_rate,
                      kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        y = self.block(x)
        return torch.cat([x, y], 1)


class Transition(nn.Sequential):
    def __init__(self, num_features):
        super(Transition, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(num_features=num_features)),
        self.add_module('relu', nn.ReLU(inplace=True)),
        self.add_module('conv', nn.Conv2d(num_features, num_features // 2, kernel_size=1, stride=1, bias=False)),
        self.add_module('poll', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """An implementation of DenseNet

    Args:
        block_config, tuple, how many dense layers in each block
        growth_rate, integer, how many filters to add each layer
        bn_size, integer, relative factor of bottleneck to growth rate

    Attributes:
        layers, sequential convolutional layers
        fc, sequential fully connected layers

    Example:
        model = DenseNet(10)
        input = torch.ones([1, 3, 32, 32])
        output = model(input)
        print(output)
    """
    def __init__(self, num_classes=10, block_config=(6, 12, 24, 16), growth_rate=32, bn_size=4):
        super(DenseNet, self).__init__()
        self.pre_layers = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm2d(num_features=64)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.dense_layers = self._make_blocks(block_config, growth_rate, bn_size)
        self.fc = nn.Linear(1024, num_classes)

    @staticmethod
    def _make_blocks(block_config, growth_rate, bn_size, init_channels=64):
        """
        Generate dense layers

        :param block_config: tuple, how many dense layers in each block
        :param growth_rate: integer, growth rate of the network
        :param bn_size: integer, relative factor of bottleneck to growth rate
        :param init_channels: integer, initial size of input channels, defaulted 64
        :return: stacked dense layers

        Example:
        blocks = DenseNet._make_blocks(block_config=[6, 12], init_channels=64, growth_rate=32, bn_size=4)
        print(blocks)
        """
        features = nn.Sequential(OrderedDict([]))
        for i, block_num in enumerate(block_config):
            dense_block = DenseNet._make_layers(block_num, init_channels, growth_rate, bn_size)
            features.add_module('dense_block%d' % (i+1), dense_block)
            init_channels += block_num * growth_rate
            if i != len(block_config) - 1:
                features.add_module('transition%d' % (i+1), Transition(init_channels))
                init_channels = init_channels // 2
        features.add_module('norm5', nn.BatchNorm2d(num_features=1024))
        return features

    @staticmethod
    def _make_layers(num_layers, in_channels, growth_rate, bn_size):
        """
        Generate dense layers

        :param num_layers: integer, how many layers to generate
        :param in_channels: integer, size of input channels
        :param growth_rate: integer, growth rate of the network
        :param bn_size: integer, relative factor of bottleneck to growth rate
        :return: stacked dense layers

        Example:
        print(DenseNet._make_layers(num_layers=6, in_channels=64, growth_rate=32, bn_size=4))
        """
        features = nn.Sequential(OrderedDict([]))
        for i in range(num_layers):
            dense_layer = DenseLayer(in_channels + i*growth_rate, bn_size, growth_rate)
            features.add_module('dense_layer%d' % (i+1), dense_layer)
        return features

    def forward(self, images):
        out = self.pre_layers(images)
        # print("size of pre layers:", out.size())
        out = self.dense_layers(out)
        # print("size of dense layers:", out.size())
        out = F.relu(out, inplace=True)
        out = F.avg_pool2d(out, kernel_size=2, stride=1)
        # print("size of avg poll:", out.size())
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
        import time
        c = Classifier('../data', force_training=False)

        test_data = Data('../data', test_batch_size=1).test_loader
        test_data = iter(test_data)
        image, label = test_data.next()

        start = time.time()
        predict = c.predict(image.to(c.device))[0]
        end = time.time()
        logger.info("Prediction of the test image ({}): {}".format(label, predict))
        logger.info("Time consumed to predict: {}".format(end - start))
    """
    def __init__(self, data_dir, model_dir='../data/densenet.pth', log_interval=50,
                 epochs=100, lr=0.01, momentum=0.5, force_training=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DenseNet(10).to(self.device)
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

    def predict(self, image):
        output = self.model(image)
        _, predicted = output.max(1)
        return predicted


if __name__ == "__main__":
    import time
    c = Classifier('../data', force_training=False)

    test_data = Data('../data', test_batch_size=1).test_loader
    test_data = iter(test_data)
    image, label = test_data.next()

    start = time.time()
    predict = c.predict(image.to(c.device))[0]
    end = time.time()
    logger.info("Prediction of the test image ({}): {}".format(label, predict))
    logger.info("Time consumed to predict: {}".format(end - start))
