import torch

x = torch.linspace(-5, -5, 30)

real_y = x * 3 + 2

weight = torch.randn(1).requires_grad_(True)
bias = torch.randn(1).requires_grad_(True)

opt = torch.optim.SGD(params=[weight, bias], lr=1e-1)

loss_func = torch.nn.MSELoss()

for epoch in range(100):
	pred_y = x * weight + bias
	loss = loss_func(pred_y, real_y)
	opt.zero_grad()
	loss.backward()
	opt.step()

	print('Epoch{}: loss = {}, weight = {}, bias = {}'.format(epoch, loss.data.numpy(), weight.data.numpy(), bias.data.numpy()))

