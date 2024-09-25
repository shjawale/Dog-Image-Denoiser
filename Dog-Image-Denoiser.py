#!/usr/bin/env python
# coding: utf-8

# ## CS224 - Winter 2024 - HW 3: Dog Denoiser

# ### Import libraries, data, viz data
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import SGD  # Our chosen optimizer
from torch.utils.data import DataLoader, TensorDataset
from nets import UNet

def show_image(xs):
    """Display a list of CIFAR-10 images in a table.
    Images may be flattened or unflattened.
    Assumes floats in range [0,1] representing colors"""
    xs = xs.cpu()
    n = len(xs)
    fig, axs = plt.subplots(1, n)
    for i, ax in enumerate(axs):
        x = xs[i].reshape(3, 32, 32).moveaxis(0, -1)
        x = torch.clamp(x, 0., 1.)
        ax.imshow(x)
        ax.axis('off')
    return fig

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# datasets are small enough to be loaded on GPU usually we leave on CPU and only put the training batch on GPU
dogs_train, dogs_val = torch.load('data.pt', map_location=device)
val_loader = DataLoader(TensorDataset(dogs_val), batch_size=50, drop_last=True)

def test_model(model, val_loader=val_loader):
    """Code to test MSE on validation data."""
    criterion = nn.MSELoss()
    model.eval()
    mse = 0.
    with torch.no_grad():
        for x, in val_loader:
            x_noise = x + 0.1 * torch.randn_like(x)
            x_hat = model(x_noise).view(x.shape)
            mse += criterion(x, x_hat) / len(val_loader)
    return mse

print("Example images of CIFAR-10 dogs")
fig = show_image(dogs_train[:5])


# ### Linear denoiser
# We will use PCA to define a linear denoiser.
# First, we encode the data into a latent factor using the top 500 principal components, then we decode to recover a (denoised) version of the image that lies on the linear subspace spanned by the first 500 components.

class LinearDenoiser(nn.Module):
    """Denoise by projecting onto linear subspace spanned by top principal components."""
    def __init__(self, d=500):
        super(LinearDenoiser, self).__init__()
        self.d = d  # Number of principal components to use
        # We won't use backprop on this model, so you can initialize/store parameters however you like
        self.mean = torch.zeros(1)
        self.U = torch.zeros(1)   #3072, 3072)
        self.S = torch.zeros(1)   #1, 3072)

    def forward(self, x):
        x = x.flatten(start_dim=1)  # Flatten images to vectors
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def encode(self, x):
        # encode x into low-d latent space
        lowD = 500
        x = x - self.mean
        z = torch.mm(x, self.U[:, :lowD])
        return z

    def decode(self, z):
        # linearly decode back to x-space
        x_hat = torch.mm(z, self.U[:, :self.d].t()) + self.mean
        return x_hat

    def fit(self, x):
        # Use PCA to get the parameters
        # Don't forget to center the data and store mean for reconstruction.
        # Use SVD to get eigenvectors of covariance, like I did in class
        x = x.flatten(start_dim=1)
        self.mean = x.mean(dim=0)
        n = len(x)  # number of images
        x = (x - self.mean)
        cov_matrix = torch.mm(x.t(), x) / n
        self.U, self.S, U_copy = torch.svd(cov_matrix)  # U_copy is identical to U, because cov is symmetric
        print(self.U.shape, self.S.shape)

linear_model = LinearDenoiser()
linear_model.fit(dogs_train)


# ### Train U-Net 
# Use the included U-Net, and train it to denoise images by minimizing the Mean Square Error loss (nn.MSELoss) between images and reconstructions from a noisy version of the image.
# Use a noise standard deviation of 0.1.
# Train with SGD.

# Train the UNet.
n_epochs = 60
unetmodel = UNet()
criterion = nn.MSELoss()
model = unetmodel.to(device)
optimizer = SGD(model.parameters(), lr=0.1)

dogs = dogs_train[:].to(device)
noisy_dogs_train = dogs + 0.1 * torch.randn_like(dogs)  # Done on CPU to avoid MPS bug!

for _ in range(n_epochs):
    model.train()                   # back to train mode
    optimizer.zero_grad()
    out = model(noisy_dogs_train)
    loss = criterion(dogs, out)     # forward pass
    loss.backward()                 # backward pass
    optimizer.step()                # implements the gradient descent step
    print(f'Train Loss: {loss.item():.3f}')


# ###  Results
# Prints out denoised images and validation loss using my trained UNet and fitted Linear model.
dogs = dogs_val[:5].cpu()
noisy_dogs = dogs + 0.1 * torch.randn_like(dogs)  # Done on CPU to avoid MPS bug!
with torch.no_grad():
    linear_denoise = linear_model(noisy_dogs.to(device)).cpu()
    unet_denoise = model(noisy_dogs.to(device)).cpu()

print("Original images")
display(show_image(dogs))
print("Noisy images")
display(show_image(noisy_dogs))
print("Linear denoising")
display(show_image(linear_denoise))
print("UNet denoising")
display(show_image(unet_denoise))

linear_mse = test_model(linear_model)
unet_mse = test_model(model)
print(f"Linear model Val MSE: {linear_mse:.4f}")
print(f"UNet Val MSE: {unet_mse:.4f}")
