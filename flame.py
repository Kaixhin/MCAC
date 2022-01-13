from matplotlib import cm
import numpy as np
import torch
from torch import multiprocessing as mp
from torch.nn.functional import softmax
from torchvision.utils import save_image


def V_0(x, y, c, f, p_1, p_2, p_3, p_4):  # Linear
  return x, y

def V_1(x, y, c, f, p_1, p_2, p_3, p_4):  # Sinusoidal
  return np.sin(x), np.sin(y)

def V_2(x, y, c, f, p_1, p_2, p_3, p_4):  # Spherical
  r = np.sqrt(x ** 2 + y ** 2)
  return 1 / ((r ** 2) * x + 1e-8), 1 / ((r ** 2) * y + 1e-8)

def V_3(x, y, c, f, p_1, p_2, p_3, p_4):  # Swirl
  r = np.sqrt(x ** 2 + y ** 2)
  return x * np.sin(r ** 2) - y * np.cos(r ** 2), x * np.cos(r ** 2) + y * np.sin(r ** 2)

def V_4(x, y, c, f, p_1, p_2, p_3, p_4):  # Horseshoe
  r = np.sqrt(x ** 2 + y ** 2)
  return 1 / (r * ((x - y) * x + y) + 1e-8), 1 / (r * 2 * x * y + 1e-8)

def V_17(x, y, c, f, p_1, p_2, p_3, p_4):  # Popcorn
  return x + c * np.sin(np.tan(3 * y)), y + f * np.sin(np.tan(3 * x))

def V_24(x, y, c, f, p_1, p_2, p_3, p_4):  # PDJ
  return np.sin(p_1 * y) - np.cos(p_2 * x), np.sin(p_3 * x) - np.cos(p_4 * y)


F = [V_0, V_1, V_2, V_3, V_4, V_17, V_24]
height, width = 64, 64
cmap = cm.get_cmap('nipy_spectral')
gamma = 4
batch_size = 64
imgs = torch.zeros(batch_size, 3, height, width).share_memory_()


# Parameters
batch_weights = softmax(torch.randn(batch_size, len(F)), dim=1).share_memory_()  # Function weights
batch_params = torch.randn(batch_size, len(F), 10).share_memory_()  # Function params
batch_colours = torch.rand(batch_size, len(F)).share_memory_()  # Function colours

def create_fractal_image(batch_i):
  weights, params, colours = batch_weights[batch_i], batch_params[batch_i], batch_colours[batch_i]

  img = torch.zeros(4, height, width)  # RGBA
  (x, y), colour = np.random.uniform(-1, 1, 2), np.random.uniform(0, 1)  # Get initial coordinates and colour
  for iteration in range(20000):
    i = np.random.multinomial(1, weights).nonzero()[0][0]  # Pick a random function
    a, b, c, d, e, f, p_1, p_2, p_3, p_4 = params[i]  # Get associated parameters
    x, y = F[i](a * x + b * y + c, d * x + e * y + f, c, f, p_1, p_2, p_3, p_4)  # Get new coordinates
    colour = (colour + colours[i].item()) / 2  # Blend colour with function colour
    if iteration >= 20:  # Plot points after the first 20 iterations
      img_x, img_y = int(width // 2 * (x + 1)), int(height // 2 * (y + 1)) 
      if 0 <= img_x < width and 0 <= img_y < height:  # Plot point if in range
        img[:3, img_y, img_x] = (img[:3, img_y, img_x] + torch.as_tensor(cmap(colour)[:3])) / 2  # Merge colours
        img[3, img_y, img_x] += 1  # Increment count
  img[3] = img[3].log() / img[3].max().log()  # Use log-density display
  img = img[:3] * (img[3] ** (1 / gamma))  # Use gamma correction
  imgs[batch_i] = img


pool = mp.Pool(8)
pool.map(create_fractal_image, range(batch_size))
pool.close()
pool.join()
save_image(imgs, 'fractal.png')
