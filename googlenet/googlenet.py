"""An implementation of GoogLeNet to classify CIFAR-10 dataset

The model is slightly tailored in several ways. The original network
was designed for dataset with much higher resolution while images in CIFAR-10
only have resolution of 32 * 32. Due to resolution discrepancy, this version
of GoogLeNet has smaller kernel sizes and stride paces in one way and the fully
connected layers are reduced to only one layer in another.

With being trained for 100 epochs, the accuracy of this network is around 80%.

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
    https://arxiv.org/pdf/1409.4842.pdf

TODO:

"""
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Data(object):
    """Prepare CIFAR-10 data for training and testing.

    Args:
        data_dir (str): directory to save downloaded data
        train_batch_size (int): batch size of training data, defaulted 64
        test_batch_size (int): batch size of testing data, defaulted 1024

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
    def __init__(self, data_dir, train_batch_size=64, test_batch_size=1024):
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


class InceptionModule(nn.Module):
    """An implementation of Inception module

    Example:
        inputs = torch.ones([2, 3, 32, 32])
        inception = InceptionModule(in_channels=3, n1=64, n3red=96, n3=128, n5red=16, n5=32, npool=32)
        outputs = inception(inputs)
        print("Size of Output:", outputs.size())
    """
    def __init__(self, in_channels, n1, n3red, n3, n5red, n5, npool):
        super(InceptionModule, self).__init__()
        # branch conv1
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=n1, kernel_size=1),
            nn.BatchNorm2d(num_features=n1),
            nn.ReLU(inplace=True),
        )

        # branch conv3
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=n3red, kernel_size=1),
            nn.BatchNorm2d(num_features=n3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n3red, out_channels=n3, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=n3),
            nn.ReLU(inplace=True),
        )

        # branch conv5
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=n5red, kernel_size=1),
            nn.BatchNorm2d(num_features=n5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=n5red, out_channels=n5, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=n5),
            nn.ReLU(inplace=True),
        )

        # branch conv5
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=npool, kernel_size=1),
            nn.BatchNorm2d(num_features=npool),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(nn.Module):
    """An implementation of GoogLeNet

    Args:

    Attributes:
        features, sequential convolutional  layers
        fc, sequential fully connected layers

    Example:
        model = GoogLeNet(10)
        input = torch.ones([1, 3, 32, 32])
        output = model(input)
        print(output)
    """
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),
        )

        self.a3 = InceptionModule(in_channels=192, n1=64, n3red=96, n3=128, n5red=16, n5=32, npool=32)
        self.b3 = InceptionModule(in_channels=256, n1=128, n3red=128, n3=192, n5red=32, n5=96, npool=64)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 480 * 16 * 16

        self.a4 = InceptionModule(in_channels=480, n1=192, n3red=96, n3=208, n5red=16, n5=48, npool=64)
        self.b4 = InceptionModule(in_channels=512, n1=160, n3red=112, n3=224, n5red=24, n5=64, npool=64)
        self.c4 = InceptionModule(in_channels=512, n1=128, n3red=128, n3=256, n5red=24, n5=64, npool=64)
        self.d4 = InceptionModule(in_channels=512, n1=112, n3red=144, n3=288, n5red=32, n5=64, npool=64)
        self.e4 = InceptionModule(in_channels=528, n1=256, n3red=160, n3=320, n5red=32, n5=128, npool=128)
        self.max_pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 832 * 8 * 8

        self.a5 = InceptionModule(in_channels=832, n1=256, n3red=160, n3=320, n5red=32, n5=128, npool=128)
        self.b5 = InceptionModule(in_channels=832, n1=384, n3red=192, n3=384, n5red=48, n5=128, npool=128)
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, images):
        out = self.pre_layers(images)
        out = self.a3(out)
        out = self.b3(out)
        out = self.max_pool3(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.max_pool4(out)
        out = self.a5(out)
        out = self.b5(out)
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
    def __init__(self, data_dir, model_dir='../data/googlenet.pth', log_interval=50,
                 epochs=100, lr=0.01, momentum=0.5, force_training=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GoogLeNet(10).to(self.device)
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
