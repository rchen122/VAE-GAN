import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
	def __init__(self, in_channels, latent_dim):
		super().__init__()
		self.in_channels = in_channels
		self.latent_dim = latent_dim

		self.downsample = nn.Sequential(
			nn.Conv2d(self.in_channels, 32, 5, 2, 2), # 64 x 16 x 16
			nn.ReLU(),
			nn.Conv2d(32, 64, 5, 2, 2), # 128 x 8 x 8
			nn.ReLU(),
			nn.Conv2d(64, 128, 5, 2, 2), # 256 x 4 x 4
			nn.ReLU()
		)
		self.fc = nn.Linear(128 * 4 * 4, 1024)
		self.bnorm = nn.BatchNorm1d(1024)
		self.relu = nn.ReLU()

		self.mu = nn.Linear(1024, latent_dim)
		self.logvar = nn.Linear(1024, latent_dim)
		
	def forward(self, x):
		B = x.shape[0]

		x = self.downsample(x)
		x = x.view(B, -1)
		x = self.fc(x)
		x = self.bnorm(x)
		x = self.relu(x)

		mu = self.mu(x)
		logvar = self.logvar(x)
		logvar = torch.clamp(logvar, min=-10, max=10)
		return mu, logvar


class Decoder(nn.Module):
	def __init__(self, latent_dim): #Batch_size, latent_dim
		super().__init__()
		self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
		self.bnorm = nn.BatchNorm1d(128 * 4 * 4)

		self.relu = nn.ReLU()
		self.upsample = nn.Sequential(
			nn.ConvTranspose2d(128, 64, 5, 2, 2, output_padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 32, 5, 2, 2, output_padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(32, 16, 5, 2, 2, output_padding=1),
			nn.ReLU(),
			nn.Conv2d(16, 3, 3, 1, 1),
		)
		self.tanh = nn.Tanh()

	def forward(self, x):
		B = x.shape[0]
		x = self.fc(x)
		x = self.bnorm(x)
		x = self.relu(x)
		x = x.reshape(B, 128, 4, 4)
		x = self.upsample(x)
		x = self.tanh(x)
		return x
	

class VAE(nn.Module):
	def __init__(self, in_channel, latent_dim):
		super().__init__()
		self.in_channel = in_channel
		self.latent_dim = latent_dim
		self.encoder = Encoder(in_channel, latent_dim)
		self.decoder = Decoder(latent_dim)

	def encode(self, x):
		return self.encoder(x)
	
	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		Z = mu + std * eps
		return Z
	
	def decode(self, x):
		return self.decoder(x)

	def forward(self, x):
		mu, logvar = self.encoder(x)
		Z = self.reparameterize(mu, logvar)
		skip = F.interpolate(F.avg_pool2d(x, kernel_size=8), 
						size=(32, 32), 
						mode='bilinear')
		x_recon = self.decode(Z) + 0.5 * skip

		return mu, logvar, x_recon