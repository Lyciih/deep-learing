import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time

# 設定訓練參數
BATCH_SIZE = 10
INPUT_SIZE = 2
LEARN_RATE = 0.0001
MOMENTUM = 0.9
EPOCH = 10
MODE = 0  # 0:從頭開始訓練 1:載入checkpoint繼續訓練 2:載入模型繼續訓練 3:單純使用模型
LOAD_MODEL_PATH = "./model/model-2023-06-18 10_26_42.pt"
LOAD_CHECKPOINT_PATH = "./checkpoint/checkpoint-2023-06-18 10_29_15.pt"
USE_GPU = 1

DEVICE = "cpu"
if USE_GPU == 1:
    check_gpu = torch.cuda.is_available()
    if check_gpu == 1:
        print('可以使用gpu')
        DEVICE = "cuda:0"
    else:
        print('不能使用gpu')

# 取得本次執行時間
train_time_temp = time.localtime()

# torch.set_default_dtype(torch.float) 改變全域預設類型，必要時再使用


# 設定訓練資料所在的路徑
train_in_path = "./data/train_in.csv"
train_out_path = "./data/train_out.csv"
test_in_path = "./data/test_in.csv"

# 用 pandas 讀取資料
train_in_data = pd.read_csv(train_in_path)
train_out_data = pd.read_csv(train_out_path)
test_in_data = pd.read_csv(test_in_path)

# 將要使用的行取出，建立 numpy 的陣列
train = np.array(
    train_in_data[['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'x8', 'y8']])
label = np.array(train_out_data["Label"])
test = np.array(
    test_in_data[['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'x8', 'y8']])

# ----------------------------------------------------------------------------------------------------------------------------#
# 轉換所有資料類型為浮點數，並建立空陣列等下用來分割資料
tensor_all_train = torch.tensor(train, dtype=torch.float)
tensor_train_cat = []
tensor_valid_temp = []

tensor_all_label = torch.tensor(label, dtype=torch.float)
tensor_train_label_temp = []
tensor_valid_label_temp = []

tensor_test = torch.tensor(test, dtype=torch.float)
tensor_test = tensor_test.view(-1, 8, 2).to(DEVICE)

# 分割訓練資料，每6筆中拿1筆資料出來當驗證集，並整理格式
tensor_all_train_split = torch.split(tensor_all_train, 6)
tensor_all_label_split = torch.split(tensor_all_label, 6)

for block in tensor_all_train_split:
    tensor_train_cat.append(block[0])
    tensor_train_cat.append(block[1])
    tensor_train_cat.append(block[2])
    tensor_train_cat.append(block[3])
    tensor_train_cat.append(block[4])
    tensor_valid_temp.append(block[5])

tensor_train = tensor_train_cat[0].unsqueeze(0)
for i in range(1, len(tensor_train_cat)):
    tensor_train = torch.cat((tensor_train, tensor_train_cat[i].unsqueeze(0)), dim=0)
tensor_train = tensor_train.view(-1, 8, 2).to(DEVICE)

tensor_valid = tensor_valid_temp[0].unsqueeze(0)
for i in range(1, len(tensor_valid_temp)):
    tensor_valid = torch.cat((tensor_valid, tensor_valid_temp[i].unsqueeze(0)), dim=0)
tensor_valid = tensor_valid.view(-1, 8, 2).to(DEVICE)

for block in tensor_all_label_split:
    tensor_train_label_temp.append(block[0])
    tensor_train_label_temp.append(block[1])
    tensor_train_label_temp.append(block[2])
    tensor_train_label_temp.append(block[3])
    tensor_train_label_temp.append(block[4])
    tensor_valid_label_temp.append(block[5])

tensor_train_label = tensor_train_label_temp[0].unsqueeze(0)
for i in range(1, len(tensor_train_label_temp)):
    tensor_train_label = torch.cat((tensor_train_label, tensor_train_label_temp[i].unsqueeze(0)), dim=0)
tensor_train_label = tensor_train_label.to(DEVICE)

tensor_valid_label = tensor_valid_label_temp


# ----------------------------------------------------------------------------------------------------------------------------#

# 設計資料集
class Train_Data(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.n = data.__len__()

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return self.n


# 設計神經網路
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=200,
            num_layers=2,
            batch_first=True,
            dropout=0.05
        )

        self.out = nn.Linear(200, 10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out


if MODE == 0 or MODE == 1 or MODE == 2:
    # 實例化資料集
    train = Train_Data(tensor_train, tensor_train_label)
    # 實例化載入器
    train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)

    if MODE == 0:  # MODE 0:從頭開始訓練
        # 實例化神經網路
        rnn = RNN()
        rnn.to(DEVICE)
    elif MODE == 1:  # MODE 1:載入checkpoint繼續訓練
        rnn = torch.load(LOAD_CHECKPOINT_PATH)
        rnn.train()
        rnn.to(DEVICE)
    else:  # MODE 2:載入模型繼續訓練
        rnn = torch.load(LOAD_MODEL_PATH)
        rnn.train()
        rnn.to(DEVICE)

    # 實例化優化器，使用 SGD
    optimizer = torch.optim.SGD(rnn.parameters(), lr=LEARN_RATE, momentum=MOMENTUM)
    # 實例化 loss function ，使用 CrossEntropy
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    # 開始訓練
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = x
            b_y = y
            output = rnn(b_x)
            loss = loss_fn(output, b_y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 每完成一個 EPOCH 就做一次驗證
        valid_output = rnn(tensor_valid)
        pred_y = torch.max(valid_output, 1)[1].data.cpu().numpy().squeeze()
        accuracy = sum(pred_y == tensor_valid_label) / len(pred_y)
        print('Epoch: ', epoch, '| train loss: %.4f' % loss, '| valid accuracy: %.2f' % accuracy)
        # 每完成一個 EPOCH 就覆蓋一次checkpoint，中途當機或暫停時可以從斷掉的點繼續，不用全部重來
        torch.save(
            rnn,
            './checkpoint/{}.pt'
            .format(time.strftime("checkpoint-%Y-%m-%d %H_%M_%S", train_time_temp)))

    # 將 10 筆驗證資料丟入模型預測
    valid_output = rnn(tensor_valid[:10].view(-1, 8, 2))
    pred_y = torch.max(valid_output, 1)[1].data.cpu().numpy().squeeze()

    # 將 10 筆驗證資料的預測結果整理成陣列顯示
    pred_temp = []
    for item in pred_y[:10]:
        pred_temp.append(item.item())
    print(pred_temp, 'prediction number')

    # 將 10 筆驗證資料的正確答案整理成陣列顯示
    real_temp = []
    for item in tensor_valid_label[:10]:
        real_temp.append(item.int().item())
    print(real_temp, 'real number')

    # 將 test 資料輸入模型，取得預測結果
    test_output = rnn(tensor_test)
    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()

    # 將預測結果存成csv檔，以開始時間做為檔名
    test_save = pd.DataFrame(pred_y, range(1, len(pred_y) + 1))
    test_save.columns = ['Label']
    test_save.to_csv('./output/{}.csv'
                     .format(time.strftime("prediction-%Y-%m-%d %H_%M_%S", train_time_temp)),
                     index_label="Serial No."
                     )

    # 儲存訓練完的model
    torch.save(
        rnn,
        './model/{}.pt'
        .format(time.strftime("model-%Y-%m-%d %H_%M_%S", train_time_temp)))

else:  # MODE 3:使用模式
    model = torch.load(LOAD_MODEL_PATH)
    model.eval()
    model.to(DEVICE)

    # 將 test 資料輸入模型，取得預測結果
    test_output = model(tensor_test)
    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()

    # 將預測結果存成csv檔，以開始時間做為檔名
    test_save = pd.DataFrame(pred_y, range(1, len(pred_y) + 1))
    test_save.columns = ['Label']
    test_save.to_csv('./output/{}.csv'
                     .format(time.strftime("prediction-%Y-%m-%d %H_%M_%S", train_time_temp)),
                     index_label="Serial No."
                     )

