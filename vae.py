import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, 16)
        self.layer4m = nn.Linear(16, z_dim + 4)
        self.layer4v = nn.Linear(16, z_dim + 4)
        self.nonlinear = nn.GELU()

    def forward(self, x, *args, **kwargs):
        x = self.layer1(x)
        x = self.nonlinear(x)
        x = self.layer2(x)
        x = self.nonlinear(x)
        x = self.layer3(x)
        x = self.nonlinear(x)
        m = self.layer4m(x)
        v = F.softplus(self.layer4v(x))
        return m, v


class Decoder(nn.Module):
    def __init__(self, h, w, z_dim, alpha=0.1):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.layer1 = nn.Linear(self.z_dim + 2, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        ys = torch.linspace(-1, 1, h)
        xs = torch.linspace(-1, 1, w)
        uv = torch.stack(torch.meshgrid(ys, xs), dim=2).unsqueeze(0)
        uv = torch.cat([uv, torch.ones_like(uv[..., 0:1])], dim=3)
        self.register_buffer('uv', uv)
        self.apply_transform = True
        self.alpha = alpha
        self.nonlinear = nn.GELU()

    def forward(self, latent, *args, **kwargs):
        z, p = latent[:, :self.z_dim], latent[:, self.z_dim:]
        n, c = z.shape
        z = rearrange(z, 'n c -> n () () c', n=n, c=c)

        _, h, w, _ = self.uv.shape
        uv = self.uv.expand(z.size(0), -1, -1, -1)
        if self.apply_transform:
            # p = (scale, theta, tx, ty)
            p *= self.alpha
            p_mat = torch.eye(3, device=p.device).unsqueeze(0).expand(n, -1, -1).clone()
            p_mat[:, 0, 0] = torch.cos(p[:, 1])
            p_mat[:, 0, 1] = -torch.sin(p[:, 1])
            p_mat[:, 0, 2] = p[:, 2]
            p_mat[:, 1, 0] = torch.sin(p[:, 1])
            p_mat[:, 1, 1] = torch.cos(p[:, 1])
            p_mat[:, 1, 2] = p[:, 3]
            uv = torch.einsum('bhwi,bij->bhwj', uv, p_mat)

        z = z.expand(-1, h, w, -1)
        uvz = torch.cat([uv[:, :, :, 0:2], z], axis=-1)

        x = self.layer1(uvz)
        x = self.nonlinear(x)
        x = self.layer2(x)
        x = self.nonlinear(x)
        x = self.layer3(x)
        x = torch.sigmoid(x)
        x = rearrange(x, 'n h w () -> n (h w)', n=n, h=h, w=w)
        return x


class Predictor(nn.Module):
    def __init__(self, z_dim):
        super(Predictor, self).__init__()
        self.layer1 = nn.Linear(z_dim, 256)
        self.layer2 = nn.Linear(256, 10)
        self.nonlinear = nn.GELU()

    def forward(self, x, *args, **kwargs):
        x = self.layer1(x)
        x = self.nonlinear(x)
        x = self.layer2(x)
        return x


# torch.log(0)によるnanを防ぐ
def torch_log(x):
    return torch.log(torch.clamp(x, min=1e-10))


class VAE(nn.Module):
    def __init__(self, h, w, z_dim):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(h, w, z_dim)
        self.predictor = Predictor(z_dim)

    def loss(self, x, y, mean, std):
        KL = torch.mean(-0.5 * torch.sum(1 + torch_log(std**2) - mean**2 - std**2, dim=1))
        reconstruction = -1 * (x * torch_log(y) + (1 - x) * torch_log(1 - y)).sum(-1).mean()
        return KL, reconstruction

    def _sample(self, mean, std):
        epsilon = torch.randn_like(mean)
        return mean + std * epsilon

    def forward(self, x, *args, **kwargs):
        mean, std = self.encoder(x)
        latent = self._sample(mean, std)
        y = self.decoder(latent)

        pred = self.predictor(latent[:, :self.z_dim])
        KL, reconstruction = self.loss(x, y, mean, std)
        return y, KL, reconstruction, pred