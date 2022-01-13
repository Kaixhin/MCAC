from collections import deque
from copy import deepcopy
import numpy as np
import torch
from torchvision.utils import save_image

from models import Discriminator, FractalFlameGenerator


def evaluate_mc(generator, discriminator, threshold, num_evaluations):
  with torch.no_grad():
    img = generator()
    mc_satisfied = discriminator(img).std().item() > threshold
    if mc_satisfied: save_image(img, f'results/{num_evaluations}.png')
    return mc_satisfied


# TODO: Implement
def evolve_seed_genomes(rand_pop, num_seeds):
  return rand_pop


def remove_oldest(viable_pop, num_removals):
  youngest = np.argsort([s.age for s in viable_pop])
  return deque([viable_pop[i] for i in youngest[:len(viable_pop) - num_removals]])


def reproduce(parents, mutation_rate):
  children = []
  for parent in parents:
    child = deepcopy(parent)
    child.age = 0
    if isinstance(child, Discriminator): child.usage = 0
    for parameter in child.parameters():
      parameter.data.add_(mutation_rate * torch.randn_like(parameter.data))
    if isinstance(child, FractalFlameGenerator): child.constrain_parameters()
    children.append(child)
  return children
