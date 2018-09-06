import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, input):
        out = F.relu(self.fc1(input))
        return self.fc2(out)


if __name__ == "__main__":
    from data import Data
    data = Data(data_dir='../data', batch_size=100, transform=transforms.ToTensor())
    net = NeuralNet(28*28, 500, 10)
    element = data.train_dataset[0][0].reshape(-1, 28 * 28)
    out = net.forward(element)
    print("output of the given input:")
    print('\t', out)
    print("label of the given input = {}".format(data.train_dataset[0][1]))