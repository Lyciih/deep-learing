import torch

class ProgressBoard(torch.HyperParameters):
	def __init__(self, xlabel=None, ylabel=None, xlim=None,
			ylim=None, xscale='linear', yscale='linear',
			ls=['-', '--', '-.', ':'], color=['c0', 'c1', 'c2', 'c3'],
			fig=None, axes=None, figsize=(3.5, 2.5), display=True):
		self.save_hyperparameters()

	def draw(self, x, y, label, every_n=1):
		raise NotImplemented

board = d21.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
	board.draw(x, np.sin(x), 'sin', every_n=2)
	board.draw(x, np.cos(x), 'cos', every_n=10)
