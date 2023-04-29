import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time

torch.set_default_dtype(torch.float64)

# csv檔的路徑
data_path = "C:\\Users\\Lyciih\\Desktop\\test\\train.csv"
test_path = "C:\\Users\\Lyciih\\Desktop\\test\\test.csv"


# 使用 read_csv() 函數來讀檔
source_data = pd.read_csv(data_path)
x1 = source_data["x1"]
x2 = source_data["x2"]
y = source_data["y"]


test_data = pd.read_csv(test_path)
test_x1 = test_data["x1"]
test_x2 = test_data["x2"]

source_data = torch.tensor(source_data.values)
test_data = torch.tensor(test_data.values)


class MyData(Dataset):

    def __init__(self, data):
        self.x1 = data[:, 1]
        self.x2 = data[:, 2]
        self.y = data[:, 3]
        self.n = data.shape[0]

    def __getitem__(self, idx):
        return torch.tensor([self.x1[idx], self.x2[idx]]), self.y[idx]

    def __len__(self):
        return self.n


class TestData(Dataset):

    def __init__(self, data):
        self.x1 = data[:, 1]
        self.x2 = data[:, 2]
        self.n = data.shape[0]

    def __getitem__(self, idx):
        return torch.tensor([self.x1[idx], self.x2[idx]])

    def __len__(self):
        return self.n


class Network(nn.Module):
    def __init__(self, in_dim, hid, output):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid)
        self.linear2 = nn.Linear(hid, output)

    def forward(self, input_array):
        h = F.relu(self.linear1(input_array))
        y_pred = self.linear2(h)
        return y_pred


BATCH_SIZE = 10
LEARN_RATE = 0.04
MOMENTUM = 0.9
EPOCHS = 1
LAYER_1 = 40

train = MyData(source_data)
loss_fn = nn.MSELoss()
MyNetwork = Network(2, 40, 1)
optimizer = torch.optim.SGD(MyNetwork.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)
train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)

finish_time = time.strftime("%Y/%m/%d     \n%H:%M:%S", time.localtime())
plt.figure(figsize=(18, 9))
plt.suptitle('{}\n'
             '\n'
             'LEARN_RATE : {}\n'
             'MOMENTUM  : {}\n'
             'EPOCHS        : {}\n'
             'BATCH_SIZE  : {}\n'
             'layer 1          : {}\n'
             .format(finish_time, LEARN_RATE, MOMENTUM, EPOCHS, BATCH_SIZE, LAYER_1),
             x=0.65, y=0.45, ha='left', size=25)

an = plt.subplot(2, 2, 1, projection='3d')
an.set_title("train")
an.scatter(x1, x2, y, c='b', marker='.')
an.set_xlim(-1, 1)
an.set_ylim(-1, 1)
an.set_zlim(-1, 2)
an.set_xlabel('X1')
an.set_ylabel('X2')
an.set_zlabel('Y')
an.view_init(elev=40., azim=-30, roll=0)


te = plt.subplot(2, 2, 2, projection='3d')
te.set_title("test")
te.set_xlim(-1, 1)
te.set_ylim(-1, 1)
te.set_zlim(-1, 2)
te.set_xlabel('X1')
te.set_ylabel('X2')
te.set_zlabel('Y')
te.view_init(elev=40., azim=-30, roll=0)


lo = plt.subplot(2, 2, 3)
lo.set_title("loss")
lo.set_xlabel('epoch times')
lo.set_ylabel('loss')




i = 0
loss = 0
loss_list = []
i_list = []
for epoch in range(EPOCHS):
    i = i + 1
    for x1_x2, label in train_loader:
        t_p = MyNetwork(x1_x2)
        loss = loss_fn(torch.flatten(t_p), label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_list.append(loss.detach())
    i_list.append(i)
    print(i, loss.detach())
lo.plot(i_list, loss_list, 'b')


test = TestData(test_data)
test_loader = DataLoader(dataset=test, batch_size=BATCH_SIZE, shuffle=True)


i = 0
for epoch in range(1):
    for x1_x2 in test_loader:
        i = i + 1
        t_p = MyNetwork(x1_x2)
        te.scatter(x1_x2[:, 0].numpy(), x1_x2[:, 1].numpy(), torch.flatten(t_p).detach().numpy(), c='r', marker='.')

plt.savefig('C:\\Users\\Lyciih\\Desktop\\test\\report\\{}.png'.format(time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())))
plt.show()
