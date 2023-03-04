import torch

#產生 0 ~ 99 的數列
dataset = torch.tensor([[i, 0, 0, 0, 0] for i in range(100)])

#將數列每九筆分成一塊(若無法被 9 整除，最後一塊會比較短)
#此函數回傳的是由分割出來的每一塊所組成的串列，之後要各別操作時需要用[index]
train_set = dataset.split(9)

#得到 train_set 第 0 組的 0 ~ 7 個元素
train_temp = train_set[0][:8]

#根據 train_set 的長度產生去掉 0 的數列
for i in range(1,len(train_set)):

	#將第 0 組與之後每一組的 0 ~ 7 元素拼接在一起
	train_temp = torch.cat((train_temp, train_set[i][:8]), dim = 0)

#印出訓練資料
print("\n訓練資料")
print(train_temp)

#從 dataset 的第 8 個元素開始，每 9 個取一個做為測試資料
test = dataset[8::9]

#印出測試資料
print("\n測試資料")
print(test)
