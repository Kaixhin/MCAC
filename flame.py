from matplotlib import cm
import numpy as np
from scipy.special import softmax
import torch
from torchvision.utils import save_image


def V_0(x, y, c, f, p_1, p_2, p_3, p_4):  # Linear
  return x, y

def V_1(x, y, c, f, p_1, p_2, p_3, p_4):  # Sinusoidal
  return np.sin(x), np.sin(y)

def V_2(x, y, c, f, p_1, p_2, p_3, p_4):  # Spherical
  r = np.sqrt(x ** 2 + y ** 2)
  return 1 / (r ** 2) * x, 1 / (r ** 2) * y

def V_3(x, y, c, f, p_1, p_2, p_3, p_4):  # Swirl
  r = np.sqrt(x ** 2 + y ** 2)
  return x * np.sin(r ** 2) - y * np.cos(r ** 2), x * np.cos(r ** 2) + y * np.sin(r ** 2)

def V_4(x, y, c, f, p_1, p_2, p_3, p_4):  # Horseshoe
  r = np.sqrt(x ** 2 + y ** 2)
  return 1 / r * ((x - y) * x + y), 1 / r * 2 * x * y

def V_17(x, y, c, f, p_1, p_2, p_3, p_4):  # Popcorn
  return x + c * np.sin(np.tan(3 * y)), y + f * np.sin(np.tan(3 * x))

def V_24(x, y, c, f, p_1, p_2, p_3, p_4):  # PDJ
  return np.sin(p_1 * y) - np.cos(p_2 * x), np.sin(p_3 * x) - np.cos(p_4 * y)


F = [V_0, V_1, V_2, V_3, V_4, V_17, V_24]
cmap = cm.get_cmap('nipy_spectral')

height, width = 64, 64
imgs = []
for _ in range(16):
  weights = softmax(np.random.randn(len(F)))  # Function weights
  params = []
  for _ in range(len(F)):
    params.append(np.random.uniform(-1, 1, 10))  # Function params
  colours = np.random.uniform(0, 1, len(F))  # Function colours

  img = torch.zeros(3, height, width)
  (x, y), colour = np.random.uniform(-1, 1, 2), np.random.uniform(0, 1)  # Get initial coordinates and colour
  for iteration in range(10000):
    i = np.random.multinomial(1, weights).nonzero()[0][0]  # Pick a random function
    a, b, c, d, e, f, p_1, p_2, p_3, p_4 = params[i]  # Get associated parameters
    x, y = F[i](a * x + b * y + c, d * x + e * y + f, c, f, p_1, p_2, p_3, p_4)  # Get new coordinates
    colour = (colour + colours[i]) / 2  # Blend colour with function colour
    if iteration >= 20:  # Plot points after the first 20 iterations
      img_x, img_y = int(width // 2 * (x + 1)), int(height // 2 * (y + 1)) 
      if 0 <= img_x < width and 0 <= img_y < height: img[:, img_y, img_x] = torch.as_tensor(cmap(colour)[:3])  # Plot point if in range
  imgs.append(img)

save_image(torch.stack(imgs), 'fractal.png')
