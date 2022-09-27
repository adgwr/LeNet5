from layers import MyConv, MyLinear, MyAvgPool
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np


# 实现LeNet5网络
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.__conv1 = MyConv(1, 6, 5, 1)
        self.__avg_pool = MyAvgPool(2, 2)
        self.__conv2 = MyConv(6, 16, 5, 1)
        self.__fc1 = MyLinear(400, 120)
        self.__fc2 = MyLinear(120, 84)
        self.__fc3 = MyLinear(84, 10)
        # 保存最佳参数
        self.__best_conv1 = None
        self.__best_conv2 = None
        self.__best_fc1 = None
        self.__best_fc2 = None
        self.__best_fc3 = None
        self.__best_loss = None

        self.__batch_size = 0
        self.__train_loss = []
        self.__test_loss = []
        self.__train_acc = []
        self.__test_acc = []

    def forward(self, x):
        x = torch.sigmoid(self.__avg_pool(self.__conv1(x)))
        x = torch.sigmoid(self.__avg_pool(self.__conv2(x)))
        x = x.view(-1, 400)
        x = torch.sigmoid(self.__fc1(x))
        x = torch.sigmoid(self.__fc2(x))
        x = self.__fc3(x)
        return x

    def predict(self, test_dataset):
        test_data_loader = DataLoader(test_dataset, batch_size=self.__batch_size)
        pred_list = []
        acc = 0.0
        with torch.no_grad():
            for idx, (test_data, test_target) in enumerate(test_data_loader):
                test_data = test_data.cuda()
                y_score = self.forward(test_data)
                _, y_pred = torch.max(y_score, 1)
                pred_list += y_pred.cpu().numpy().tolist()
                acc += accuracy_score(test_target, y_pred.cpu()) * len(test_data.cpu()) / len(test_dataset)
        return pred_list, acc

    def __update_best_para(self, loss):
        if loss < self.__best_loss:
            self.__best_loss = loss
            self.__best_conv1 = copy.deepcopy(self.__conv1)
            self.__best_conv2 = copy.deepcopy(self.__conv2)
            self.__best_fc1 = copy.deepcopy(self.__fc1)
            self.__best_fc2 = copy.deepcopy(self.__fc2)
            self.__best_fc3 = copy.deepcopy(self.__fc3)

    def __evaluate_test(self, test_dataset):
        with torch.no_grad():
            test_data_loader = DataLoader(test_dataset, batch_size=self.__batch_size)
            test_loss = 0
            for _, (test_data, test_target) in enumerate(test_data_loader):
                test_data, test_target = test_data.cuda(), test_target.cuda()
                y_test_score = self.forward(test_data)
                cross_entropy = nn.CrossEntropyLoss()
                batch_test_loss = cross_entropy(y_test_score, test_target)
                test_loss += float(batch_test_loss) * len(test_data) / len(test_dataset)
            self.__test_loss.append(test_loss)

    def __plot(self):
        plt.figure(1)
        train_len = len(self.__train_loss)
        test_len = len(self.__test_loss)
        plt.plot(range(train_len), self.__train_loss, label='train loss')
        plt.plot(np.linspace(0, train_len, test_len), self.__test_loss, label='test loss')
        plt.legend()
        plt.xlabel('iter times')
        plt.ylabel('loss')
        plt.savefig('loss.png')
        plt.show()
        plt.figure(2)
        plt.plot(range(len(self.__train_acc)), self.__train_acc, label='train acc')
        plt.plot(range(len(self.__test_acc)), self.__test_acc, label='test acc')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.savefig('accuracy.png')
        plt.show()

    # 每经过iter_times_per_eval迭代，模型在测试集上评估一次
    def fit(self, train_dataset, test_dataset, optimizer,
            batch_size, epoch, iter_times_per_eval=10):
        self.__train_loss.clear()
        self.__test_loss.clear()
        self.__train_acc.clear()
        self.__test_acc.clear()
        self.__best_loss = float('inf')
        self.__batch_size = batch_size
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for i in range(epoch):
            for idx, (data, target) in enumerate(train_data_loader):
                # 正向传播，计算loss
                data, target = data.cuda(), target.cuda()
                y_score = self.forward(data)
                cross_entropy = nn.CrossEntropyLoss()
                loss = cross_entropy(y_score, target)
                self.__update_best_para(float(loss))
                self.__train_loss.append(float(loss))

                # 反向传播，梯度下降
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 在测试集上评估
                if idx % iter_times_per_eval == 0:
                    self.__evaluate_test(test_dataset)
                    print('iter {}, train loss: {}, test loss: {}'.format(idx, self.__train_loss[-1], self.__test_loss[-1]))

            _, train_acc = self.predict(train_dataset)
            _, test_acc = self.predict(test_dataset)
            self.__train_acc.append(train_acc)
            self.__test_acc.append(test_acc)
            print('Epoch {}/{}, train acc: {}, test acc: {}'.format(i + 1, epoch, train_acc, test_acc))
        self.__plot()

        # 令参数为训练过程中最小loss的参数
        self.__conv1 = self.__best_conv1
        self.__conv2 = self.__best_conv2
        self.__fc1 = self.__best_fc1
        self.__fc2 = self.__best_fc2
        self.__fc3 = self.__best_fc3