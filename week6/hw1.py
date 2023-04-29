import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd


# csv檔的路徑
data_path = "C:\\Users\\Lyciih\\Desktop\\test\\train.csv"
test_path = "C:\\Users\\Lyciih\\Desktop\\test\\test.csv"

# 使用 read_csv() 函數來讀檔
source_data = pd.read_csv(data_path)
x1 = source_data["x1"]
x2 = source_data["x2"]
y = source_data["y"]

"""
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(x1, x2, y, c='b', marker='.')

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 2)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.view_init(elev=20., azim=-35, roll=0)

"""



test_data = pd.read_csv(test_path)
test_x1 = test_data["x1"]
test_x2 = test_data["x2"]


# 將資料轉為tensor型式
source_data = torch.tensor(source_data.values)
test_data = torch.tensor(test_data.values)


class MyData(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        self.x1 = self.data[idx][1]
        self.x2 = self.data[idx][2]
        self.y = self.data[idx][3]
        return self.x1, self.x2, self.y

    def __len__(self):
        return self.data.shape[0]


class TestData(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        self.x1 = self.data[idx][1]
        self.x2 = self.data[idx][2]
        return self.x1, self.x2

    def __len__(self):
        return self.data.shape[0]


# 實例化一個MyData，並把讀到的數據丟進去
hw1 = MyData(source_data)


# 產生隨機的 w1 , w2 (以 2 * 1 矩陣的形式)
w = torch.tensor([0], dtype=torch.float64, requires_grad=True)
b = torch.zeros(size=(1, 10), requires_grad=True)

# 建立 DataLoader
# 為甚麼給參數要用等號?
loader = DataLoader(dataset=hw1, batch_size=10, shuffle=True)
loss = nn.MSELoss()
optimizer = torch.optim.SGD(params=[w], lr=0.5)


x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.torch.tanh(x)
y.backward(torch.ones_like(x), retain_graph=True)
plt.plot(x.detach(), y.detach(), 'b')
plt.show()
"""
# 此處的 range 代表全部資料要跑幾次 假設總資料筆數是 8000 ， 1 就是8000 2 就是 1600 3就是 2400
time = 0
for epoch in range(10):
    # x 是用來接收 loader 回傳的資料 (x 只是一個名字，可以隨便取)
    for x1, x2, label in loader:
        time = time + 1
        optimizer.zero_grad()
        t1 = w * np.sin((np.sqrt((x1.pow(2) + x2.pow(2))))) + b
        output = loss(torch.flatten(t1), label)
        output.backward()
        optimizer.step()
        print(time, torch.flatten(w).data, output.item())


# 實例化一個TestData，並把讀到的數據丟進去
test1 = TestData(test_data)
test_loader = DataLoader(dataset=test1, batch_size=10, shuffle=True)


time = 0
for epoch in range(1):
    for x1, x2 in test_loader:
        time = time + 1
        
        t1 = w * np.sin(np.sqrt((x1.pow(2) + x2.pow(2)))) + b
        ax.scatter(x1.numpy(), x2.numpy(), t1.detach().numpy(), c='r', marker='.')
        print(time, torch.flatten(w).data, t1)

plt.show()
print(w, b)


class MyData(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        x1 = self.data[idx][1]
        x2 = self.data[idx][2]
        y = self.data[idx][3]
        return torch.tensor([x1, x2]), y

    def __len__(self):
        return self.data.shape[0]
        

# 此處的 range 代表全部資料要跑幾次 假設總資料筆數是 8000 ， 1 就是8000 2 就是 1600 3就是 2400
time = 0
for epoch in range(1):
    # x 是用來接收 loader 回傳的資料 (x 只是一個名字，可以隨便取)
    for x1_x2, label in loader:
        optimizer.zero_grad()
        time = time + 1
        t1 = x1_x2 @ w
        output = loss(torch.flatten(t1), label)
        output.backward()
        optimizer.step()
        ax.scatter(x1_x2[:, 0].numpy(), x1_x2[:, 1].numpy(), t1.detach().numpy(), c='r', marker='.')
        print(time, torch.flatten(w).data, output.item())

plt.show()



# 以下兩種句型都可以操作 __getitem__ 函數
# print(hw1[0])
# print(hw1.__getitem__(0))

# 可以操作 __len__ 函數
# print(hw1.__len__())
# print(len(hw1))

# a = 1.0
# b = 2.0
# c = torch.tensor([a, b])
# d = torch.tensor([[a], [b]])

# e = c @ d
# print(c)
# print(d)
# print(e)

# 取得資料總共有幾筆
data_size = source_data.shape[0]
# 取得資料總共有幾個欄位
data_column = source_data.shape[1]
"""