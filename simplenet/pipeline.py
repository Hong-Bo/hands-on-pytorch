from .neuralnet import NeuralNet
from .data import Data
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms


class Pipeline(object):
    def __init__(self, input_size, hidden_size, output_size,
                 data_dir, transform, batch_size,
                 log_interval, epochs, lr=0.01, momentum=0.5,
                 save_model=False, load_model=None):

        self.model = NeuralNet(input_size, hidden_size, output_size)
        self.data = Data(data_dir, transform, batch_size)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.log_interval = log_interval
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_model = save_model
        self.load_model = load_model

    def train(self, epoch):
        self.model.train()
        for batch_idx, (images, target) in enumerate(self.data.train_loader):
            images = images.reshape(-1, 28 * 28)
            images, target = images.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(images)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
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
            for data, target in self.data.test_loader:
                data = data.reshape(-1, 28 * 28)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()

                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        len_data = len(self.data.test_loader.dataset)
        test_loss /= len_data
        print("\nTest results: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len_data, 100. * correct/len_data
        ))

    def run(self):
        if self.load_model is not None:
            self.model.load_state_dict(torch.load('simplenet.ckpt'))
            self.test()
            return True

        for epoch in range(1, self.epochs + 1):
            self.train(epoch)
            self.test()

        if self.save_model:
            torch.save(self.model.state_dict(), 'simplenet.ckpt')


if __name__ == "__main__":
    pipe = Pipeline(
        input_size=28 * 28, hidden_size=500, output_size=10,
        data_dir='./data', batch_size=100, transform=transforms.ToTensor(),
        log_interval=50, epochs=10, save_model=True, load_model=True
        )
    pipe.run()