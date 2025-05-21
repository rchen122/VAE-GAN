import torch
import torch.nn as nn
from models import simple_vae
import argparse
import yaml
import torchvision
# from torch.optim.lr_scheduler import CosineAnnealingLR
from util import *
import time

def train_loop(config, load_model):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	torch.autograd.set_detect_anomaly(True)

	train_config = config['train']
	model_config = config['model']
	epochs = train_config['epochs']

	# Initialize Model Parameters
	in_channel = model_config['in_channel']
	latent_dim = model_config['latent_dim']

	vae = simple_vae.VAE(in_channel, latent_dim).to(device)
	def weights_init(m):
		if isinstance(m, nn.ConvTranspose2d):
			nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
			
	vae.decoder.apply(weights_init)


	vae_opt = torch.optim.Adam([
		{'params': vae.encoder.parameters(), 'lr': 3e-4},
		{'params': vae.decoder.parameters(), 'lr': 3e-4}
	], weight_decay=1e-5)


	bce = nn.BCEWithLogitsLoss()
	mse = nn.MSELoss()

	_, train_loader = trainLoader(train_config['batch_size'])

	if load_model:
		checkpoint = torch.load(load_model)

		vae.load_state_dict(checkpoint['vae'])
		epoch_start = checkpoint['epoch']
		best_loss = checkpoint['loss']
	else:
		best_loss = torch.inf
		epoch_start = 1


	print("Starting Training Loop")
	for epoch in range(epoch_start, epochs + 1):
		running_vae = 0
		running_kl_loss = 0
		running_logvar = 0
		print(f"Running Epoch {epoch}")
		# recon_min, recon_max, recon_mean = 0, 0, 0
		for images, _ in train_loader:
			images = images.to(device)
			B = images.size(0)

			# ENCODER/DECODER
			vae_opt.zero_grad()
			mu, logvar, x_recon = vae(images)
			l_dis = mse(x_recon, images)

			beta = 0.01
			kl_div, l_prior = loss_fn(epoch, beta, mu, logvar, device)
			l_vae = l_prior + l_dis

			l_vae.backward() 
			# print(f"l_prior: {l_prior.item()}, mu: {mu.mean()}, {mu.std()}, logvar: {logvar.mean()}, {logvar.std()}")
			# # Gradient Debugging
			# total_grad = 0
			# for name, param in vae.named_parameters():
			# 	if param.grad is not None:
			# 		grad_mean = param.grad.abs().mean().item()
			# 		print(f"{name} grad: {grad_mean:.6f}")
			# 		total_grad += grad_mean
			
			# print(f"Total gradient flow: {total_grad:.6f}")

			torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=10.0)
			vae_opt.step()
			running_vae += l_vae.item()
			running_kl_loss += kl_div
			running_logvar += logvar.mean().item()

			# recon_min += x_recon.min()
			# recon_max += x_recon.max()
			# recon_mean += x_recon.mean()

		# print(f"Recon Min: {recon_min/len(train_loader):.3f}, Recon Max: {recon_max/len(train_loader):.3f}, Recon Mean: {recon_mean/len(train_loader):.3f}")
		print(f"KL_div: {running_kl_loss/len(train_loader)}, Logvar: {running_logvar/len(train_loader)}")
		print(f"Epoch [{epoch}/{epochs}], Vae Loss: {running_vae/len(train_loader):.4f}")
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
