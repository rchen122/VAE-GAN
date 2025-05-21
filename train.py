import torch
import torch.nn as nn
from models import vae_gan
import argparse
import yaml
# from torch.optim.lr_scheduler import CosineAnnealingLR
from util import *
import time


def train_loop(config, load_model):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")
	torch.autograd.set_detect_anomaly(True)

	train_config = config['train']
	model_config = config['model']
	epochs = train_config['epochs']

	# Initialize Model Parameters
	in_channel = model_config['in_channel']
	latent_dim = model_config['latent_dim']
	vae = vae_gan.VAE(in_channel, latent_dim)
	discrim = vae_gan.Discriminator(in_channel, latent_dim).to(device)
	vae_opt = torch.optim.Adam(vae.parameters(), lr=1e-4)
	dis_opt = torch.optim.Adam(discrim.parameters(), lr=1e-4)
	bce = nn.BCEWithLogitsLoss()
	mse = nn.MSELoss()

	_, train_loader = trainLoader(train_config['batch_size'])

	if load_model:
		checkpoint = torch.load(load_model)
		vae.load_state_dict(checkpoint['vae'])
		discrim.load_state_dict(checkpoint['discriminator'])
		epoch_start = checkpoint['epoch']
		best_loss = checkpoint['loss']
	else:
		best_loss = torch.inf
		epoch_start = 1


	gan_flag = False # GAN Training
	print("Starting Training Loop")
	for epoch in range(epoch_start, epochs + 1):
		running_disc = 0
		running_enc = 0
		running_dec = 0

		recon_min = 0
		recon_max = 0
		recon_mean = 0
		for images, _ in train_loader:
			images = images.to(device)
			B = images.size(0)

			# DISCRIMINATOR
			if epoch > 10 and gan_flag:
				dis_opt.zero_grad()
				with torch.no_grad():
					mu, logvar = vae.encoder(images)
					Z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
					x_recon = vae.decoder(Z)
					Z_p = torch.randn(B, latent_dim).to(device)
					X_p = vae.decoder(Z_p)

				D_real = discrim(images)
				D_recon = discrim(x_recon.detach())
				D_fake = discrim(X_p.detach())

				real_label = torch.empty_like(D_real).uniform_(0.8, 1.0)
				fake_label = torch.empty_like(D_fake).uniform_(0.0,0.2)
				l_gan_dis =(bce(D_real, real_label) + 
							bce(D_recon, fake_label) + 
							bce(D_fake, fake_label)) / 3.0 
				l_gan_dis.backward()
				dis_opt.step()
			else: 
				l_gan_dis = torch.tensor(0.0, device=device)

			# ENCODER/DECODER
			vae.zero_grad()

			mu, logvar = vae.encoder(images) 
			std = torch.exp(0.5 * logvar) 
			eps = torch.randn_like(std)
			Z = mu + std * eps
			x_recon = vae.decoder(Z) # VAE DECODING
			
			if epoch > 10 and gan_flag:	
				feat_fake = discrim.features_forward(x_recon) #feature matching loss	
				feat_real = discrim.features_forward(images).detach()
				l_dis = mse(feat_fake, feat_real)	

				Z_p = torch.randn(B, latent_dim).to(device)
				X_p = vae.decoder(Z_p) # GAN GENERATOR
				D_recon = discrim(x_recon.detach())
				D_fake = discrim(X_p.detach())
				gan_label = torch.empty_like(D_fake).uniform_(0.8, 1.0)
				l_gan = (bce(D_recon, gan_label) + 
						bce(D_fake, gan_label)) / 2.0
			else:
				l_gan = torch.tensor(0.0, device=device)
				l_dis = mse(x_recon, images)
			
			beta = 0.1
			l_prior = beta * kl_divergence(mu, logvar)
			l_decoder = l_dis + 0.2 * l_gan 
			l_encoder = l_prior + l_dis
			l_vae = l_decoder + l_encoder

			l_vae.backward()
			vae.step()

			running_enc += l_encoder.item()
			running_dec += l_decoder.item()
			running_disc += l_gan_dis.item()

			recon_min += x_recon.min()
			recon_max += x_recon.max()
			recon_mean += x_recon.mean()
		print(f"Recon Min: {recon_min/len(train_loader):.3f}, Recon Max: {recon_max/len(train_loader):.3f}, Recon Mean: {recon_mean/len(train_loader):.3f}")
		print(f"Epoch [{epoch}/{epochs}], Encoder Loss: {running_enc/len(train_loader):.4f}, Decoder Loss: {running_dec/len(train_loader):.4f}, Discrim Loss: {running_disc/len(train_loader):.4f}")
		total_loss = l_encoder.item() + l_decoder.item() + l_gan_dis.item()
		total_loss = l_vae.item()
		if total_loss < best_loss:
			best_loss = total_loss
			torch.save({
				'vae': vae.state_dict(),
				'epoch': epoch,
				'loss': best_loss,
			}, train_config['dst'] + 'best_checkpoint.pth')
		if epoch % 10 == 0: # every 25 runs:
			torch.save({
				'vae': vae.state_dict(),
				'epoch': epoch,
				'loss': best_loss,
			}, train_config['dst'] + f'{epoch}_checkpoint.pth')
		torch.save({
			'vae': vae.state_dict(),
			'epoch': epoch,
			'loss': best_loss,
			}, train_config['dst'] + 'last_checkpoint.pth')
		
		log_samples(epoch, vae, train_loader, device, 4)
		if epoch % 2 == 1:
			visualize_latent_space(epoch, vae, train_loader, device, "pca", 10000)




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default='config.yaml', help='Path to config file')
	parser.add_argument('--load-model', type=str, required=False, help="Loads existing model")
	args = parser.parse_args()
	load_model = args.load_model
	with open(args.config, "r") as file:
		config = yaml.safe_load(file)
	train_loop(config, load_model)
