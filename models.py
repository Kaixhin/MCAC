from typing import Optional
import torch
from torch import jit, nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


def _weight_init(module):
  if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
    nn.init.normal_(module.weight, 0, 0.02)
    if module.bias is not None: nn.init.constant_(module.bias, 0)


def adaptive_instance_norm_2d(x, y):
  ys, yb = y.chunk(2, dim=1)
  x = (x - x.mean(dim=(-1, -2), keepdim=True)) / (x.std(dim=(-1, -2), keepdim=True) + 1e-8)  # Normalisation
  x = ys.view(-1, x.size(1), 1, 1) * x + yb.view(-1, x.size(1), 1, 1)
  return x


def relu_sin_tanh(x):
  x1, x2, x3 = x.chunk(3, dim=-1)
  return torch.cat([F.relu(x1), torch.sin(x2), torch.tanh(x3)], dim=-1)


class CoordConv2d(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, height, width, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    super().__init__(in_channels + 2, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
    self.height, self.width = height, width
    x_grid, y_grid = torch.meshgrid(torch.linspace(-1, 1, width), torch.linspace(-1, 1, height))
    self.register_buffer('coordinates', torch.stack([x_grid, y_grid]).unsqueeze(dim=0))

  def forward(self, x):
    x = torch.cat([x, self.coordinates.expand(x.size(0), 2, self.height, self.width)], dim=1)  # Concatenate spatial embeddings
    return super().forward(x)


class SelfAttention2d(nn.Module):
  def __init__(self, in_channels, normalise=False):
    super().__init__()
    self.att_channels = in_channels // 8
    self.Wf = nn.Conv2d(in_channels, self.att_channels, 1, padding=0)
    self.Wg = nn.Conv2d(in_channels, self.att_channels, 1, padding=0)
    self.Wh = nn.Conv2d(in_channels, in_channels, 1, padding=0)
    if normalise: self.Wf, self.Wg, self.Wh = spectral_norm(self.Wf), spectral_norm(self.Wg), spectral_norm(self.Wh)
    self.gamma = nn.Parameter(torch.zeros(1))  # Initialise attention at 0

  def forward(self, x):
    B, C, H, W = x.size()
    f = self.Wf(x).view(B, self.att_channels, -1).permute(0, 2, 1)  # Query
    g = self.Wg(x).view(B, self.att_channels, -1)  # Key
    h = self.Wh(x).view(B, C, -1).permute(0, 2, 1)  # Value
    beta = F.softmax(f @ g, dim=2)  # Attention
    o = (beta @ h).permute(0, 2, 1).view(B, C, H, W)
    y = self.gamma * o + x
    return y


class StyleGANBlock(nn.Module):
  def __init__(self, input_size, output_size, latent_size, upsample=True):
    super().__init__()
    if upsample:
      self.conv1 = nn.Conv2d(input_size, input_size, 3, padding=1)
    else:
      self.base = nn.Parameter(torch.randn(1, input_size, 4, 4))
    self.B1 = nn.Parameter(torch.tensor([0.01]))
    self.A1 = nn.Linear(latent_size, 2 * input_size)
    self.conv2 = nn.Conv2d(input_size, output_size, 3, padding=1)

  def forward(self, w: torch.Tensor, x: Optional[torch.Tensor]=None):
    if hasattr(self, 'base'):
      x = self.base
    elif x is not None and hasattr(self, 'conv1'):
      x = F.relu(self.conv1(F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)))
    x = x + self.B1 * torch.randn_like(x)
    x = adaptive_instance_norm_2d(x, F.relu(self.A1(w)))
    x = F.relu(self.conv2(x))
    return x


class Generator(nn.Module):
  def __init__(self):
    super().__init__()


class StyleGANGenerator(Generator):
  def __init__(self, latent_size, hidden_size=64):
    super().__init__()
    self.age = 0

    self.latent_size = latent_size
    # self.mapping = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.ELU(), nn.Linear(hidden_size, hidden_size), nn.ELU(), nn.Linear(hidden_size, hidden_size), nn.ELU(), nn.Linear(hidden_size, hidden_size))
    self.block1 = StyleGANBlock(16 * hidden_size, 8 * hidden_size, latent_size, upsample=False)  # 4x4
    self.block2 = StyleGANBlock(8 * hidden_size, 4 * hidden_size, latent_size)  # 8x8
    self.block3 = StyleGANBlock(4 * hidden_size, 2 * hidden_size, latent_size)  # 16x16
    self.block4 = StyleGANBlock(2 * hidden_size, hidden_size, latent_size)  # 32x32
    self.block5 = StyleGANBlock(hidden_size, hidden_size, latent_size)  # 64x64
    self.conv = nn.Conv2d(hidden_size, 3, 5, padding=2)
    self.apply(_weight_init)

  def forward(self, z):
    # w = self.mapping(z)
    x = self.block1(z)
    x = self.block2(z, x)
    x = self.block3(z, x)
    x = self.block4(z, x)
    x = self.block5(z, x)
    return torch.tanh(self.conv(x)) / 2 + 0.5


class CPPNGenerator(Generator):
  def __init__(self, latent_size, hidden_size=64):
    super().__init__()
    self.age = 0

    self.height, self.width = 64, 64
    x_grid, y_grid = torch.meshgrid(torch.linspace(-3, 3, self.width), torch.linspace(-3, 3, self.height))
    self.register_buffer('coordinates', torch.stack([x_grid, y_grid]).unsqueeze(dim=0))
    self.latent_size = latent_size
    self.fc1 = nn.Linear(latent_size + 2, 3 * hidden_size)
    self.fc2 = nn.Linear(3 * hidden_size, 3 * hidden_size)
    self.fc3 = nn.Linear(3 * hidden_size, 3 * hidden_size)
    self.fc4 = nn.Linear(3 * hidden_size, 3 * hidden_size)
    self.fc5 = nn.Linear(3 * hidden_size, 3)
    self.apply(_weight_init)

  def forward(self, z):
    z = torch.cat([z.expand(z.size(0), self.latent_size, self.height, self.width), self.coordinates.expand(z.size(0), 2, self.height, self.width)], dim=1).permute(0, 2, 3, 1)
    x = relu_sin_tanh(self.fc1(z))
    x = relu_sin_tanh(self.fc2(x))
    x = relu_sin_tanh(self.fc3(x))
    x = relu_sin_tanh(self.fc4(x))
    return torch.sigmoid(self.fc5(x)).permute(0, 3, 1, 2)


class Discriminator(nn.Module):
  def __init__(self, hidden_size=64):
    super().__init__()
    self.age, self.usage = 0, 0

    self.conv1 = spectral_norm(nn.Conv2d(3, hidden_size, 4, stride=2, padding=1))
    self.conv2 = spectral_norm(nn.Conv2d(hidden_size, 2 * hidden_size, 4, stride=2, padding=1))
    self.in2 = nn.InstanceNorm2d(2 * hidden_size)
    self.conv3 = spectral_norm(nn.Conv2d(2 * hidden_size, 4 * hidden_size, 4, stride=2, padding=1))
    self.in3 = nn.InstanceNorm2d(4 * hidden_size)
    # self.att3 = SelfAttention2d(4 * hidden_size, normalise=True)
    self.conv4 = spectral_norm(nn.Conv2d(4 * hidden_size, 8 * hidden_size, 4, stride=2, padding=1))
    self.in4 = nn.InstanceNorm2d(8 * hidden_size)
    self.conv5 = spectral_norm(nn.Conv2d(8 * hidden_size, 1, 4, stride=1, padding=0))
    self.apply(_weight_init)

  def forward(self, x):
    x = F.leaky_relu(self.conv1(x), 0.2)
    x = F.leaky_relu(self.in2(self.conv2(x)), 0.2)
    x = F.leaky_relu(self.in3(self.conv3(x)), 0.2)
    # x = self.att3(x)
    x = F.leaky_relu(self.in4(self.conv4(x)), 0.2)
    return self.conv5(x).view(-1)


def generate_random_population(generator, pop_size, latent_size, hidden_size):
  if generator == 'StyleGAN':
    G = StyleGANGenerator
  elif generator == 'CPPN':
    G = CPPNGenerator
  return [G(latent_size, hidden_size) for _ in range(pop_size)] + [Discriminator(hidden_size) for _ in range(pop_size)]
