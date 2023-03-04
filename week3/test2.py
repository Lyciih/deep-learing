import torch

data = torch.Tensor([[i] * 5 for i in range(100)])
test_data = data[8::9, :]
mask = (torch.arange(len(data)) % 9) < 8
train_data = data[mask, :]

print(data)
