import os
import torch
import numpy as np
import random
from dataset import MNISTDataset
from model import LeNet5


def set_random_seed(seed_value=1024):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)


if __name__ == '__main__':
    set_random_seed()
    # 获取数据集
    train_data, test_data = MNISTDataset(train=True), MNISTDataset(train=False)
    # 初始化模型并训练
    model = LeNet5()
    device = torch.device('cuda')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.fit(train_data, test_data, optimizer=optimizer, batch_size=64, epoch=100, iter_times_per_eval=100)

    # 计算模型在训练集和测试集上的准确率
    y_train_pred, train_acc = model.predict(train_data)
    print('train acc: {}'.format(train_acc))

    y_test_pred, test_acc = model.predict(test_data)
    print('test acc: {}'.format(test_acc))
