import torchvision.transforms as transforms
import torchvision
import torch


def trainLoader(batch_size):
	transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
	])

	train_dataset = torchvision.datasets.CIFAR10(root='dataset/', train=True, transform=transform, download=True)	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	return train_dataset, train_loader

def kl_divergence(mu, logvar):
	return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def log_samples(epoch, encoder, decoder, dataloader, device, num_images):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        real_batch = next(iter(dataloader))[0].to(device)[:num_images]

        mu, logvar = encoder(real_batch)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = decoder(z)

        z_random = torch.randn(num_images, encoder.latent_dim).to(device)
        generated = decoder(z_random)

        def save_grid(images, filename):
            images = (images + 1) / 2  
            grid = torchvision.utils.make_grid(images, nrow=2, padding=2)
            torchvision.utils.save_image(grid, 'output_samples/run3/' + filename)

        save_grid(real_batch, f"real_epoch_{epoch}.png")
        save_grid(recon, f"recon_epoch_{epoch}.png")
        save_grid(generated, f"gen_epoch_{epoch}.png")

    encoder.train()
    decoder.train()