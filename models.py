import torch
from torch import nn
from torch.nn import functional as F


class CoordConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, height, width, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    super().__init__()
    self.height, self.width = height, width
    self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
    x_grid, y_grid = torch.meshgrid(torch.linspace(-1, 1, width), torch.linspace(-1, 1, height))
    self.register_buffer('coordinates', torch.stack([x_grid, y_grid]).unsqueeze(dim=0))

  def forward(self, x):
    x = torch.cat([x, self.coordinates.expand(x.size(0), 2, self.height, self.width)], dim=1)  # Concatenate spatial embeddings TODO: radius?
    return self.conv(x)


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


class Generator(nn.Module):
  def __init__(self, latent_size, hidden_size=8):
    super().__init__()
    self.latent_size = latent_size
    self.z = nn.Parameter(torch.randn(latent_size))
    self.conv1 = nn.ConvTranspose2d(latent_size, 8 * hidden_size, 4, stride=1, padding=0, bias=False)
    self.conv2 = nn.ConvTranspose2d(8 * hidden_size, 4 * hidden_size, 4, stride=2, padding=1, bias=False)
    self.conv3 = nn.ConvTranspose2d(4 * hidden_size, 2 * hidden_size, 4, stride=2, padding=1, bias=False)
    self.att3 = SelfAttention2d(2 * hidden_size)
    self.conv4 = nn.ConvTranspose2d(2 * hidden_size, hidden_size, 4, stride=2, padding=1, bias=False)
    self.conv5 = nn.ConvTranspose2d(hidden_size, 1, 4, stride=2, padding=1)
  
  def forward(self):
    z = self.z.view(1, self.latent_size, 1, 1) * torch.randn(64, self.latent_size, 1, 1)
    x = F.relu(self.conv1(z))
    x = F.relu(self.conv2(x))
    x = F.relu(self.att3(self.conv3(x)))
    x = F.relu(self.conv4(x))
    return torch.sigmoid(self.conv5(x))


class Discriminator(nn.Module):
  def __init__(self, hidden_size=8):
    super().__init__()
    self.usage = 0
    self.conv1 = CoordConv2d(1, hidden_size, 4, 64, 64, stride=2, padding=1, bias=False)
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


def generate_random_population(pop_size):
  return [Generator(10) for _ in range(pop_size)] + [Discriminator() for _ in range(pop_size)]
