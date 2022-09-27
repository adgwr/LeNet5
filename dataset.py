from torch.utils.data import Dataset, DataLoader
import torchvision


# 加载MNIST数据集
class MNISTDataset(Dataset):
    def __init__(self, train=True):
        super(MNISTDataset, self).__init__()
        self.train = train
        self.data = torchvision.datasets.MNIST(root='data/',
                                               train=train,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.Pad(2),
                                                   torchvision.transforms.ToTensor()
                                               ]),
                                               download=True)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)