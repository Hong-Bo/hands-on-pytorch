import torch
import torchvision
import torchvision.transforms as transforms


class Data(object):
    def __init__(self, data_dir, transform, batch_size, test_batch_size=1000):
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

        self.train_dataset = self.dataset(True)
        self.test_dataset = self.dataset(False)
        self.train_loader = self.loader(self.train_dataset, self.batch_size)
        self.test_loader = self.loader(self.test_dataset, self.test_batch_size)

    def dataset(self, train=False):
        return torchvision.datasets.MNIST(
            root=self.data_dir, train=train, transform=self.transform, download=True
        )

    def loader(self, dataset, batch_size):
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True
        )


if __name__ == "__main__":
    data = Data(data_dir='./data', batch_size=100, transform=transforms.ToTensor())
    print("type of dataset = {}".format(type(data.train_dataset)))
    print("size of the first element = {}".format(data.train_dataset[0][0].size()))
    print("label of the first element = {}".format(data.train_dataset[0][1]))

    for i, (images, labels) in enumerate(data.train_loader):
        print("images = {}".format((images.size())))
        print("reshaped images = {}".format((images.reshape(-1, 28*28).size())))
        print("labels = {}".format((labels.size())))
        break