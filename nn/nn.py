"""Training a simple neural network to classify MNIST data

Example:
    # Classifying a MNIST image using this module

    # 1. load the Classifier from this module
    from nn import Classifier
    c = Classifier('../data', force_training=False)
    test_data = c.data.dataset(False)

    # 2. Make a prediction
    import random
    i = random.randint(1, len(test_data))
    image, label = test_data[i][0], test_data[i][1]
    print("Predicted of {}th test image ({}):".format(i, label), c.predict(image))

Attributes:

TODO:
    1. logging
    2. gitignore
    3. differentiate deployment & training

"""
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms


class Data(object):
    """Prepare MNIST data for training and testing.

    Args:
        data_dir (str): directory to save downloaded data
        train_batch_size (int): batch size of training data, defaulted 100
        test_batch_size (int): batch size of testing data, defaulted 1000

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
    def __init__(self, data_dir, train_batch_size=100, test_batch_size=1000):
        self.data_dir = data_dir
        self.transform = transforms.ToTensor()

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


class NN(nn.Module):
    """A simple neural network with two fully connected layers

    Args:
        input_size (int): number of neurons in the input layer
        hidden_size (int): number of neurons in the hidden layer
        output_size (int): number of neurons in the output layer

    Attributes:

    Example:
        model = NN(28*28, 500, 10)
        input = torch.ones(1, 28*28)
        output = model.forward(input)
        print(output)
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, input):
        out = F.relu(self.fc1(input))
        return self.fc2(out)


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
        print("Predicted of {}th test image ({}):".format(i, label), c.predict(image))
        end = time.time()
        print("Time consumed to predict:", end - start)
    """
    def __init__(self, data_dir, model_dir='../data/nn.pth', log_interval=50,
                 epochs=20, lr=0.01, momentum=0.5, force_training=False):
        self.model = NN(input_size=28*28, hidden_size=500, output_size=10)
        self.data = Data(data_dir)
        self.log_interval = log_interval
        self.epochs = epochs
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = self.load_model(model_dir, force_training)

    def load_model(self, model_dir, force_training):
        if os.path.exists(model_dir) and not force_training:
            self.model.load_state_dict(torch.load(model_dir))
        else:
            print("No pre-trained model is found. Start training from scratch")
            for epoch in range(1, self.epochs + 1):
                self.train(epoch)
                self.test()
            torch.save(self.model.state_dict(), model_dir)
        self.model.eval()
        return True

    def train(self, epoch):
        self.model.train()
        for batch_idx, (images, labels) in enumerate(self.data.train_loader):
            images = images.reshape(-1, 28 * 28)
            images, target = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            self.optimizer.step()
            if (batch_idx+1) % self.log_interval == 0:
                print(
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
                images = images.reshape(-1, 28 * 28)
                output = self.model(images)
                test_loss += F.cross_entropy(output, labels, reduction='sum').item()

                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()

        len_data = len(self.data.test_loader.dataset)
        test_loss /= len_data
        print("\nTest results: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len_data, 100. * correct/len_data
        ))

    def predict(self, image):
        output = self.model(image.reshape(-1, 28*28))
        _, predicted = output.max(1)
        return predicted


if __name__ == "__main__":
    import time
    import random
    c = Classifier('../data', force_training=False)
    test_data = c.data.dataset(False)
    i = random.randint(1, len(test_data))
    image, label = test_data[i][0], test_data[i][1]
    start = time.time()
    print("Predicted of {}th test image ({}):".format(i, label), c.predict(image))
    end = time.time()
    print("Time consumed to predict:", end - start)

