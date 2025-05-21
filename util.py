import torchvision.transforms as transforms
import torchvision
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

def loss_fn(epoch, beta, mu, logvar, device):
    beta = min(0.1, 0.01 * epoch)
    kl_div = kl_divergence(mu, logvar)
    l_prior = beta * kl_div

    # if epoch > 10: 
    #     l_prior = beta * kl_divergence(mu, logvar)
    # else:
    #     l_prior = torch.tensor(0.0, device=device)
    return kl_div, l_prior

def visualize_latent_space(epoch, model, dataloader, device, method='pca', max_samples=10000):

    model.eval()
    mu_list = []
    label_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            mu, _ = model.encode(images)  # mu: [batch_size, latent_dim]
            mu_list.append(mu.cpu())
            label_list.append(labels)

            if len(torch.cat(mu_list)) > max_samples:
                break

    mu_all = torch.cat(mu_list)[:max_samples]
    labels_all = torch.cat(label_list)[:max_samples]

    if method == 'pca':
        projector = PCA(n_components=2)
    elif method == 'tsne':
        projector = TSNE(n_components=2, perplexity=30, n_iter=1000, init='pca')
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    z_2d = projector.fit_transform(mu_all)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels_all, cmap='tab10', alpha=0.7, s=10)
    plt.colorbar(scatter, label="Class Label")
    plt.title(f"Latent Space ({method.upper()} projection of μ)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.savefig(f"output_samples/run3/graphs/latent_space_{epoch}.jpg")
    # plt.show()

    model.train()

def log_samples(epoch, vae, dataloader, device, num_images):
	vae.eval()

	with torch.no_grad():
		real_batch = next(iter(dataloader))[0].to(device)[:num_images]

		mu, logvar, recon = vae(real_batch)

		z_random = torch.randn(num_images, vae.latent_dim).to(device)
		generated = vae.decode(z_random)

		def save_grid(images, filename):
			images = (images + 1) / 2  
			grid = torchvision.utils.make_grid(images, nrow=2, padding=2)
			torchvision.utils.save_image(grid, 'output_samples/run3/' + filename)

		# save_grid(real_batch, f"real_epoch_{epoch}.png")
		save_grid(recon, f"recon_epoch_{epoch}.png")
		save_grid(generated, f"gen_epoch_{epoch}.png")

	vae.train()