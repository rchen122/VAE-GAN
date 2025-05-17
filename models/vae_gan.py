import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
	def __init__(self, in_channels, latent_dim):
		super().__init__()
		self.in_channels = in_channels
		self.downsample = nn.Sequential(
			nn.Conv2d(self.in_channels, 64, 5, 2, 2), # 64 x 32 x 32
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 128, 5, 2, 2), # 128 x 16 x 16
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 256, 5, 2, 2), # 256 x 8 x 8
			nn.BatchNorm2d(256),
			nn.ReLU()
		)
		self.fc = nn.Linear(256 * 8 * 8, 2048)
		self.bnorm = nn.BatchNorm1d(2048)
		self.relu = nn.ReLU()

		self.mu = nn.Linear(2048, latent_dim)
		self.logvar = nn.Linear(2048, latent_dim)
		
	def forward(self, x):
		B = x.shape[0]

		x = self.downsample(x)
		x = x.view(B, -1)
		x = self.fc(x)
		x = self.bnorm(x)
		x = self.relu(x)

		mu = self.mu(x)
		logvar = self.logvar(x)
		return mu, logvar


class Decoder(nn.Module):
	def __init__(self, latent_dim): #Batch_size, latent_dim
		super().__init__()
		self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
		self.bnorm = nn.BatchNorm1d(256 * 8 * 8)
		self.relu = nn.ReLU()
		self.upsample = nn.Sequential(
			nn.ConvTranspose2d(256, 128, 5, 2, 2, output_padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.ConvTranspose2d(128, 32, 5, 2, 2, output_padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.ConvTranspose2d(32, 3, 5, 2, 2, output_padding=1),
		)
		self.tanh = nn.Tanh()

	def forward(self, x):
		B = x.shape[0]
		x = self.fc(x)
		x = self.bnorm(x)
		x = self.relu(x)
		x = x.reshape(B, 256, 8, 8)
		x = self.upsample(x)
		x = self.tanh(x)
		return x
	

class Discriminator(nn.Module):
	def __init__(self, in_channels, latent_dim):
		super().__init__()
		self.in_channels = in_channels
		self.conv1 = nn.Conv2d(self.in_channels, 32, 5, 1, 2) # 32 x 64 x 64
		self.relu = nn.ReLU()
		self.downsample = nn.Sequential(
			nn.Conv2d(32, 128, 5, 2, 2), # 128 x 32 x 32
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 256, 5, 2, 2), # 256 x 16 x 16
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 256, 5, 2, 2), # 256 x 8 x 8
			nn.BatchNorm2d(256),
			nn.ReLU()
		)
		self.fc1 = nn.Linear(256 * 8 * 8, 512)
		self.bnorm = nn.BatchNorm1d(512)
		self.fc2 = nn.Linear(512, 1)
		self.sig = nn.Sigmoid()
		
	def forward(self, x):
		B = x.shape[0]
		x = self.conv1(x)
		x = self.relu(x)

		x = self.downsample(x)

		x = x.view(B, -1)
		x = self.fc1(x)
		x = self.bnorm(x)
		x = self.relu(x)

		x = self.fc2(x)
		x = self.sig(x)
		return x