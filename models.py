import torch
from torch import nn
from torch import multiprocessing as mp
from torch.nn import functional as F

from flame import create_fractal_image


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
    x_grid, y_grid = torch.meshgrid(torch.linspace(-1, 1, width), torch.linspace(-1, 1, height), indexing='ij')
    self.register_buffer('coordinates', torch.stack([x_grid, y_grid]).unsqueeze(dim=0))

  def forward(self, x):
    x = torch.cat([x, self.coordinates.expand(x.size(0), 2, self.height, self.width)], dim=1)  # Concatenate spatial embeddings
    return super().forward(x)


class SelfAttention2d(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.att_channels = in_channels // 8
    self.Wf = nn.Conv2d(in_channels, self.att_channels, 1, padding=0)
    self.Wg = nn.Conv2d(in_channels, self.att_channels, 1, padding=0)
    self.Wh = nn.Conv2d(in_channels, in_channels, 1, padding=0)
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
  def __init__(self, hidden_size, upsample=True):
    super().__init__()
    if upsample:
      self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
    else:
      self.base = nn.Parameter(torch.randn(64, hidden_size, 4, 4))
    self.B1 = nn.Parameter(torch.tensor([0.01]))
    self.A1 = nn.Linear(hidden_size, 2 * hidden_size)
    self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
    self.B2 = nn.Parameter(torch.tensor([0.01]))
    self.A2 = nn.Linear(hidden_size, 2 * hidden_size)

  def forward(self, x=None, w=None):
    if hasattr(self, 'base'):
      x = self.base
    else:
      x = F.relu(self.conv1(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)))
    x = x + self.B1 * torch.randn_like(x)
    x = adaptive_instance_norm_2d(x, self.A1(w))
    x = F.relu(self.conv2(x))
    x = x + self.B2 * torch.randn_like(x)
    x = adaptive_instance_norm_2d(x, self.A2(w))
    return x


class Generator(nn.Module):
  def __init__(self):
    super().__init__()


class StyleGANGenerator(Generator):
  def __init__(self, latent_size=10, hidden_size=8):
    super().__init__()
    self.age = 0

    batch_size = 64
    self.latent_size = latent_size
    self.z = nn.Parameter(torch.randn(batch_size, latent_size))
    self.mapping = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.ELU(), nn.Linear(hidden_size, hidden_size), nn.ELU(), nn.Linear(hidden_size, hidden_size), nn.ELU(), nn.Linear(hidden_size, hidden_size))
    self.block1 = StyleGANBlock(hidden_size, upsample=False)  # 4x4
    self.block2 = StyleGANBlock(hidden_size)  # 8x8
    self.block3 = StyleGANBlock(hidden_size)  # 16x16
    self.block4 = StyleGANBlock(hidden_size)  # 32x32
    self.block5 = StyleGANBlock(hidden_size)  # 64x64
    self.conv = nn.Conv2d(hidden_size, 3, 5, padding=2)

  def forward(self):
    w = self.mapping(self.z)
    x = self.block1(w=w)
    x = self.block2(x, w)
    x = self.block3(x, w)
    x = self.block4(x, w)
    x = self.block5(x, w)
    return torch.sigmoid(self.conv(x))


class CPPNGenerator(Generator):
  def __init__(self, latent_size=10, hidden_size=8):
    super().__init__()
    self.age = 0

    self.height, self.width = 64, 64
    x_grid, y_grid = torch.meshgrid(torch.linspace(-3, 3, self.width), torch.linspace(-3, 3, self.height), indexing='ij')
    self.register_buffer('coordinates', torch.stack([x_grid, y_grid]).unsqueeze(dim=0))
    self.latent_size = latent_size
    self.z = nn.Parameter(torch.randn(latent_size))
    self.fc1 = nn.Linear(latent_size + 2, 3 * hidden_size)
    self.fc2 = nn.Linear(3 * hidden_size, 3 * hidden_size)
    self.fc3 = nn.Linear(3 * hidden_size, 3 * hidden_size)
    self.fc4 = nn.Linear(3 * hidden_size, 3 * hidden_size)
    self.fc5 = nn.Linear(3 * hidden_size, 3)

  def forward(self):
    batch_size = 64
    z = self.z.view(1, self.latent_size, 1, 1) * torch.randn(batch_size, self.latent_size, 1, 1)
    z = torch.cat([z.expand(batch_size, self.latent_size, self.height, self.width), self.coordinates.expand(batch_size, 2, self.height, self.width)], dim=1).permute(0, 2, 3, 1)
    x = relu_sin_tanh(self.fc1(z))
    x = relu_sin_tanh(self.fc2(x))
    x = relu_sin_tanh(self.fc3(x))
    x = relu_sin_tanh(self.fc4(x))
    return torch.sigmoid(self.fc5(x)).permute(0, 3, 1, 2)


class FractalFlameGenerator(Generator):
  def __init__(self):
    super().__init__()
    self.age = 0
    
    self.batch_size, height, width = 64, 64, 64
    self.noise_scale = 3
    self.register_buffer('imgs', torch.zeros(self.batch_size, 3, height, width).share_memory_())
    self.weights = nn.Parameter(torch.randn(1, 7).share_memory_())  # Function weights; parameterisation shared across batch
    self.params = nn.Parameter(torch.randn(1, 7, 10).share_memory_())  # Function params
    self.colours = nn.Parameter(torch.rand(1, 7).share_memory_())  # Function colours

  def forward(self):
    self.imgs.zero_()  # Reset the image canvas
    weights = F.softmax(self.weights.data * self.noise_scale * torch.rand(self.batch_size, 7), dim=1)  # Multiply by batch noise and normalise weights
    params = self.params.data * self.noise_scale * torch.rand(self.batch_size, 7, 10)  # Multiply by batch noise
    colours = torch.clamp(self.colours.data * self.noise_scale * torch.rand(self.batch_size, 7), min=0, max=1)  # Multiply by batch noise and clamp colour index to [0, 1]

    pool = mp.Pool(8)
    args = [[i] + [self.imgs, weights[i], params[i], colours[i]] for i in range(self.batch_size)]
    pool.starmap(create_fractal_image, args)
    pool.close()
    pool.join()
    return self.imgs


class Discriminator(nn.Module):
  def __init__(self, hidden_size=8):
    super().__init__()
    self.age = 0
    self.usage = 0

    self.conv1 = CoordConv2d(3, hidden_size, 4, 64, 64, stride=2, padding=1, bias=False)
    self.conv2 = nn.Conv2d(hidden_size, 2 * hidden_size, 4, stride=2, padding=1, bias=False)
    self.conv3 = nn.Conv2d(2 * hidden_size, 4 * hidden_size, 4, stride=2, padding=1, bias=False)
    self.att3 = SelfAttention2d(4 * hidden_size)
    self.conv4 = nn.Conv2d(4 * hidden_size, 8 * hidden_size, 4, stride=2, padding=1, bias=False)
    self.conv5 = nn.Conv2d(8 * hidden_size, 1, 4, stride=1, padding=0, bias=False)

  def forward(self, x):
    x = F.leaky_relu(self.conv1(x), 0.2)
    x = F.leaky_relu(self.conv2(x), 0.2)
    x = F.leaky_relu(self.att3(self.conv3(x)), 0.2)
    x = F.leaky_relu(self.conv4(x), 0.2)
    return torch.sigmoid(self.conv5(x)).view(-1)


def generate_random_population(generator, pop_size):
  if generator == 'StyleGAN':
    G = StyleGANGenerator
  elif generator == 'CPPN':
    G = CPPNGenerator
  elif generator == 'FractalFlame':
    G = FractalFlameGenerator
  return [G() for _ in range(pop_size)] + [Discriminator() for _ in range(pop_size)]
