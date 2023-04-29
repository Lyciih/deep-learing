import d2l
import torch

class Module(torch.nn.Module, d2l.HyperParameters):
	def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
		super().__init__()
		self.save_hyperparameters()
		self.board = ProgressBoard()

	def loss(self, y_hat, y):
		raise NotImplementedError

	def forward(self, X):
		assert hasattr(self, 'net'), 'Neural network is defined'
		return self.net(X)

	def plot(self, key, value, train):
		assert hasattr(self, 'trainer'), 'Train is not inited'
		self.board.xlabel = 'epoch'
		if train:
			x = self.trainer.train_batch_idx / self.train.num_train_batches
			n = self.trainer.num_train_batches / self.plot_train_per_epoch
		else:
			x = self.trainer.epoch + 1
			n = self.trainer.num_val_batches / self.plot_valid_per_epoch
		self.board.draw(x, value.to(d21.cpu()).detach().numpy(),
				('train_' if train else 'val_') + key,
				every_n = int(n))
	
	def training_step(self, batch):
		l = self.loss(self(*batch[:-1]), batch[-1])
		self.plot('loss', l, train=True)
		return l

	def validation_step(self, batch):
		l = self.loss(self(*batch[:1]), batch[-1])
		self.plot('loss', l, train=False)

	def configure_optimizers(self):
		raise NotImplementedError



class DataMoudule(d21.HyperParameters):
	def __init__(self, root='../data', num_workers=4):
		self.save_hyperparameters()

	def get_dataloader(self, train):
		raise NotImplementedError
	
	def train_dataloader(self):
		return self.get_dataloader(train=True)

	def val_dataloader(self):
		return self.get_dataloader(train=False)






class Trainer(d21.HyperParameters):
	def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
		self.save_hyperparameters()
		assert num_gpus == 0, 'NO GPU support yet'

	def prepare_data(self, data):
		self.train_dataloader = data.train_dataloader()
		self.val_dataloader = data.val_dataloader()
		self.num_train_batches = len(self.train_dataloader)
		self.num_val_batches = (len(self.val_dataloader)
					if self.val_dataloader is not None else 0)


	def prepare_model(self, model):
		model.trainer = self
		model.board.xlim = [0, self.max_epochs]
		self.model = model

	def fit(self, model, data):
		self.prepare_data(data)
		self.prepare_model(model)
		self.optim = model.configure_optimizers()
		self.epoch = 0
		self.train_batch_idx = 0
		self.val_batch_idx = 0
		for self.epoch in range(self.max_epochs):
			self.fit_epoch()

	def fit_epoch(self):
		raise NotImplementedError
