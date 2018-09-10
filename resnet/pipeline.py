import torchvision.transforms as transforms
import torch.optim as optim
import torch
import torch.nn.functional as F


class Pipeline(object):
    def __init__(self, model, data, lr, momentum, log_interval, epochs, save_model=False, load_model=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.data = data
        self.learning_rate = lr
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.log_interval = log_interval
        self.epochs = epochs
        self.save_model = save_model
        self.load_model = load_model

    def train(self, epoch):
        self.model.train()
        for batch_idx, (images, labels) in enumerate(self.data.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

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

        if (epoch + 1) % 5 == 0:
            self.learning_rate /= 3
            self.update_lr(self.optimizer, self.learning_rate)

    @staticmethod
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def test(self):
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for images, labels in self.data.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                test_loss += F.cross_entropy(output, labels, reduction='sum').item()

                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()

        len_data = len(self.data.test_loader.dataset)
        test_loss /= len_data
        print("\nTest results: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len_data, 100. * correct/len_data
        ))

    def run(self):
        if self.load_model:
            self.model.load_state_dict(torch.load('resnet.ckpt'))
            self.test()
            return True

        for epoch in range(1, self.epochs + 1):
            self.train(epoch)
            self.test()

        if self.save_model:
            torch.save(self.model.state_dict(), 'resnet.ckpt')


if __name__ == "__main__":
    from net import ResNet, ResBlock
    resnet = ResNet(ResBlock, layers=[2, 2, 2])

    from data import Data
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()
    ])
    cifar10 = Data(data_dir='../data', batch_size=100, transform=transform)

    pipe = Pipeline(resnet, cifar10, lr=0.1, momentum=0, log_interval=30, epochs=20)
    pipe.run()

