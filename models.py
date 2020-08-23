import torch
from torch import nn
from torch.nn import functional as F


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
    self.bn1 = nn.BatchNorm2d(8 * hidden_size)
    self.conv2 = nn.ConvTranspose2d(8 * hidden_size, 4 * hidden_size, 4, stride=2, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(4 * hidden_size)
    self.conv3 = nn.ConvTranspose2d(4 * hidden_size, 2 * hidden_size, 4, stride=2, padding=1, bias=False)
    self.att3 = SelfAttention2d(2 * hidden_size)
    self.bn3 = nn.BatchNorm2d(2 * hidden_size)
    self.conv4 = nn.ConvTranspose2d(2 * hidden_size, hidden_size, 4, stride=2, padding=1, bias=False)
    self.bn4 = nn.BatchNorm2d(hidden_size)  
    self.conv5 = nn.ConvTranspose2d(hidden_size, 1, 4, stride=2, padding=1)
  
  def forward(self):
    z = self.z.view(1, self.latent_size, 1, 1) * torch.randn(64, self.latent_size, 1, 1)
    x = F.relu(self.bn1(self.conv1(z)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.att3(self.conv3(x))))
    x = F.relu(self.bn4(self.conv4(x)))
    return torch.sigmoid(self.conv5(x))


class Discriminator(nn.Module):
  def __init__(self, hidden_size=8):
    super().__init__()
    self.usage = 0
    self.conv1 = nn.Conv2d(1, hidden_size, 4, stride=2, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(hidden_size)
    self.conv2 = nn.Conv2d(hidden_size, 2 * hidden_size, 4, stride=2, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(2 * hidden_size)
    self.conv3 = nn.Conv2d(2 * hidden_size, 4 * hidden_size, 4, stride=2, padding=1, bias=False)
    self.att3 = SelfAttention2d(4 * hidden_size)
    self.bn3 = nn.BatchNorm2d(4 * hidden_size)
    self.conv4 = nn.Conv2d(4 * hidden_size, 8 * hidden_size, 4, stride=2, padding=1, bias=False)
    self.bn4 = nn.BatchNorm2d(8 * hidden_size)
    self.conv5 = nn.Conv2d(8 * hidden_size, 1, 4, stride=1, padding=0, bias=False)

  def forward(self, x):
    x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
    x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
    x = F.leaky_relu(self.bn3(self.att3(self.conv3(x))), 0.2)
    x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
    return torch.sigmoid(self.conv5(x)).view(-1)


def generate_random_population(pop_size):
  return [Generator(10) for _ in range(pop_size)] + [Discriminator() for _ in range(pop_size)]
