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
        return torchvision.datasets.CIFAR10(
            root=self.data_dir, train=train, transform=self.transform, download=True
        )

    def loader(self, dataset, batch_size):
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True
        )


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()
    ])

    data = Data(data_dir='../data', batch_size=100, transform=transform)
    print("type of dataset = {}".format(type(data.train_dataset)))
    print("size of the first element = {}".format(data.train_dataset[0][0].size()))
    print("label of the first element = {}".format(data.train_dataset[0][1]))

    print("Info for batches")
    for i, (images, labels) in enumerate(data.train_loader):
        print("\t size of images = {}".format((images.size())))
        print("\t size of labels = {}".format((labels.size())))
        break
