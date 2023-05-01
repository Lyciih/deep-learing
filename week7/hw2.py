import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time

torch.set_default_dtype(torch.float64)

# csv檔的路徑
data_path = "./train.csv"
test_path = "./test.csv"


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
    def __init__(self, in_dim, hid1, hid2, hid3, hid4, output):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid1)
        self.linear2 = nn.Linear(hid1, hid2)
        self.linear3 = nn.Linear(hid2, hid3)
        self.linear4 = nn.Linear(hid3, hid4)
        self.linear5 = nn.Linear(hid4, output)

    def forward(self, input_array):
        h1 = F.relu(self.linear1(input_array))
        h2 = F.relu(self.linear2(h1))
        h3 = F.relu(self.linear3(h2))
        h4 = F.relu(self.linear4(h3))
        y_pred = F.relu(self.linear5(h4))
        return y_pred


BATCH_SIZE = 10
LEARN_RATE = 0.00002
MOMENTUM = 0.9
EPOCHS = 8000
graph = [2, 40, 40, 40, 40, 1]


train = MyData(source_data)
loss_fn = nn.MSELoss()
MyNetwork = Network(graph[0], graph[1], graph[2], graph[3], graph[4],  graph[5])
optimizer = torch.optim.SGD(MyNetwork.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)
train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)


# 紀錄訓練時間，讓之後儲存的檔名統一
train_time_temp = time.localtime()
train_time = time.strftime("%Y/%m/%d  %H:%M:%S", train_time_temp)

plt.figure(figsize=(18, 9))
plt.suptitle('{}    LEARN_RATE: {}   MOMENTUM: {}   EPOCHS: {}   BATCH_SIZE: {}'
             .format(train_time, LEARN_RATE, MOMENTUM, EPOCHS, BATCH_SIZE),
             x=0.55, y=0.1, ha='left', size=10)
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


ne = plt.subplot(2, 2, 4)
ne.set_aspect(1)
ne.set_xlim(0, 2)
ne.set_ylim(0, 1)
ne.xaxis.set_visible(False)
ne.yaxis.set_visible(False)
ne.axis('off')

i = 0
h_div = 2/(len(graph) + 1)
temp_prev = []
temp_next = []
for center in graph:
    temp_prev = temp_next
    temp_next = []
    i = i + 1
    if i == 1:
        cir_color = 'g'
        ne.text(i * h_div, 1, 'input', fontsize=10, ha='center')
    elif i == len(graph):
        cir_color = 'r'
        ne.text(i * h_div, 1, 'output', fontsize=10, ha='center')
    else:
        cir_color = 'orange'
        ne.text(i * h_div, 1, 'Layer{} [{}]'.format(i - 1, graph[i-1]), fontsize=10, ha='center')
    v_div = 1 / (center + 1)
    for v in range(center):
        v = v + 1
        temp_next.append([i * h_div, v * v_div])
        for prev_node in temp_prev:
            line = plt.Line2D((prev_node[0], i * h_div), (prev_node[1], v * v_div), color='gray', zorder=1)
            ne.add_line(line)
        cir = Circle((i * h_div, v * v_div), radius=0.02, color=cir_color, zorder=2)
        ne.add_patch(cir)






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
lo.plot(i_list, loss_list, 'y')


test = TestData(test_data)
test_loader = DataLoader(dataset=test, batch_size=BATCH_SIZE, shuffle=True)


i = 0
test_out = []
for epoch in range(1):
    for x1_x2 in test_loader:
        i = i + 1
        t_p = MyNetwork(x1_x2)
        temp = torch.flatten(t_p.detach())
        for data in temp:
            test_out.append(data.numpy())
        te.scatter(x1_x2[:, 0].numpy(), x1_x2[:, 1].numpy(), torch.flatten(t_p).detach().numpy(), c='r', marker='.')


# 儲存訓練報告
plt.savefig('./report/{}.png'.format(time.strftime("%Y-%m-%d %H_%M_%S", train_time_temp)))


# 產生測試資料結果的csv檔
test_output = pd.DataFrame(test_out, range(1, len(test) + 1))
test_output.columns.name = 'id'
test_output.columns = ['y']
test_output.to_csv('./test/{}.csv'
                   .format(time.strftime("%Y-%m-%d %H_%M_%S", train_time_temp)),
                   index_label="id"
                   )


# 儲存訓練結果
torch.save(
    MyNetwork.state_dict(),
    './parameter/{}.pt'
    .format(time.strftime("%Y-%m-%d %H_%M_%S", train_time_temp)))

# plt.show()
