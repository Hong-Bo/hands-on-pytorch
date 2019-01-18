"""Training a simple neural network to classify MNIST data

Example:
    # Classifying a MNIST image using this module

    # Load the Classifier class from this module
    from nn import Classifier
    c = Classifier('../data', force_training=False)
    # Load dataset
    test_data = c.data.dataset(False)

    # Make a prediction
    import random
    i = random.randint(1, len(test_data))
    image, label = test_data[i][0], test_data[i][1]
    print("Predicted of {}th test image ({}):".format(i, label), c.predict(image))

Reference:
    http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

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
    """Prepare MNIST data for training and testing.

    Args:
        data_dir (str): directory to save downloaded data
        train_batch_size (int): batch size of training data, defaulted 256
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
            print("original images = {}".format((images.size())))
            print("reshaped images = {}".format((images.reshape(-1, 28*28).size())))
            print("labels = {}\n{}".format(labels.size(), labels))

        2. Load data
        data = Data(data_dir='../data')
        train_data = data.dataset(train=True)
        image100, label100 = train_data[100][0], train_data[100][1]
    """
    def __init__(self, data_dir, train_batch_size=256, test_batch_size=1024):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            # transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        self.train_loader = self._loader(train=True, batch_size=train_batch_size)
        self.test_loader = self._loader(train=False, batch_size=test_batch_size)

    def _loader(self, train, batch_size):
        dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=train, transform=self.transform, download=True
        )

        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True
        )

    def dataset(self, train=True):
        return torchvision.datasets.MNIST(
            root=self.data_dir, train=train, transform=self.transform, download=True
        )


class LeNet5(nn.Module):
    """An implementation of LeNet-5

    Args:

    Attributes:
        features, sequential convolutional  layers
        fc, sequential fully connected layers

    Example:
        model = LeNet5()
        input = torch.ones(1, 28*28)
        output = model.forward(input)
        print(output)
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5)),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, img):
        feats = self.features(img)
        feats = feats.view(img.size(0), -1)
        out = self.fc(feats)
        return out


class Classifier(object):
    """To classify a MNIST image
    Args:
        data_dir (path), directory of MNIST data
        model_dir (path), directory to load and save model files
        log_interval (int), interval of logging when training model
        epochs (int), how many epochs to train the model
        lr (float), learning rate of training
        momentum (float), momentum of learning
        force_training (bool), whether to train the model even it has been trained before

    Example:
        # Predict the label of a random image
        import time
        import random
        c = Classifier('../data', force_training=False)
        test_data = c.data.dataset(False)
        i = random.randint(1, len(test_data))
        image, label = test_data[i][0], test_data[i][1]
        start = time.time()
        predict = c.predict(image.unsqueeze(0))[0]
        end = time.time()
        logger.info("Predicted of {}th test image ({}): {}".format(i, label, predict))
        logger.info("Time consumed to predict: {}".format(end - start))
    """
    def __init__(self, data_dir, model_dir='../data/lenet.pth', log_interval=50,
                 epochs=20, lr=0.01, momentum=0.5, force_training=False):
        self.model = LeNet5()
        self.data = Data(data_dir)
        self.epochs = epochs
        self.log_interval = log_interval
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = self.load_model(model_dir, force_training)

    def load_model(self, model_dir, force_training):
        if os.path.exists(model_dir) and not force_training:
            self.model.load_state_dict(torch.load(model_dir))
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
            images, target = images.to(self.device), labels.to(self.device)

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
        # Predict the label of a random image
        import time
        import random
        c = Classifier('../data', force_training=False)
        test_data = c.data.dataset(False)
        i = random.randint(1, len(test_data))
        image, label = test_data[i][0], test_data[i][1]
        start = time.time()
        predict = c.predict(image.unsqueeze(0))[0]
        end = time.time()
        logger.info("Predicted of {}th test image ({}): {}".format(i, label, predict))
        logger.info("Time consumed to predict: {}".format(end - start))
