from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from vae import VAE


batch_size = 256
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
n_epochs = 20
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Lambda(lambda x: x.view(-1))
])

trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True
)

testset = datasets.MNIST('./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False
)


vae = VAE(28, 28, z_dim=10).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

with tqdm(range(1, n_epochs+1)) as pbar:
    for epoch in pbar:
        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            y, KL, recon, pred = vae(inputs)
            loss = recon + KL + F.cross_entropy(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        correct = 0
        KL_ = 0
        recon_ = 0
        for data in testloader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                y, KL, recon, pred = vae(inputs)
                _, p = torch.topk(pred, k=1)
                correct += (p == labels).sum().item()
                KL_ += KL.item()
                recon_ += recon.item()

        correct /= len(testloader.dataset)
        KL_ /= len(testloader)
        recon_ /= len(testloader)
        
        pbar.set_description(f'[Epoch {epoch}]')
        pbar.set_postfix({'KL':f'{KL_:.3f}', 'Recon':f'{recon_:.3f}', 'TestAcc':f'{correct:.2f}%'})   

        import numpy as np


x = torch.stack([trainset[i][0] for i in range(12)]).to(device)
#x = torch.stack([testset[i][0] for i in range(12)]).to(device)

vae.decoder.apply_transform = True
with torch.no_grad():
    y, KL, recon, pred = vae(x)

vae.decoder.apply_transform = False
with torch.no_grad():
    z, KL, recon, pred = vae(x)

x = x.reshape(-1, 28, 28).cpu().numpy()
y = y.reshape(-1, 28, 28).cpu().numpy()
z = z.reshape(-1, 28, 28).cpu().numpy()

fig = plt.figure(figsize=(10, 10))
for i in range(12):
    ax = fig.add_subplot(6, 6, i+1, xticks=[], yticks=[])
    ax.imshow(x[i])
    ax = fig.add_subplot(6, 6, i+12+1, xticks=[], yticks=[])
    ax.imshow(y[i])
    ax = fig.add_subplot(6, 6, i+24+1, xticks=[], yticks=[])
    ax.imshow(z[i])

plt.show()     