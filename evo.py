from copy import deepcopy
import torch
from torchvision.utils import save_image

from models import Discriminator


I = 0


def evaluate_mc(generator, discriminator):
  global I
  I += 1
  with torch.no_grad():
    img = generator()
    mc_satisfied = discriminator(img).std().item() > 0.3
    if mc_satisfied: save_image(img, f'results/{I}.png')
    return mc_satisfied


# TODO: Implement
def evolve_seed_genomes(rand_pop, num_seeds):
  return rand_pop


def reproduce(parents):
  children = []
  for parent in parents:
    child = deepcopy(parent)
    if isinstance(child, Discriminator): child.usage = 0
    for parameter in child.parameters():
      parameter.data.add_(0.2 * torch.randn_like(parameter.data))
    children.append(child)
  return children
