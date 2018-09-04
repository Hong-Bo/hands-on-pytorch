import torch
import torch.nn.functional as F
from neuralnet import NeuralNet
from data import Data
import torchvision.transforms as transforms
import torch.optim as optim


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (images, target) in enumerate(train_loader):
        images = images.reshape(-1, 28*28)
        images, target = images.to(device), target.to(device)
        optimizer.zero_grad()  # clear all gradients manually
        output = model(images)
        loss = F.cross_entropy(output, target)  # The negative log likelihood loss
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


if __name__ == '__main__':
    data = Data(data_dir='./data', batch_size=100, transform=transforms.ToTensor())
    model = NeuralNet(input_size=28 * 28, hidden_size=500, output_size=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(10):
        train(model, device, data.train_loader, optimizer, epoch)