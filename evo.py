from collections import deque
from copy import deepcopy
import numpy as np
import torch
from torch import autograd
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA
from torchvision.utils import save_image

from models import Discriminator


def evaluate_mc(generator, discriminator, threshold, num_evaluations):
  with torch.no_grad():
    img = generator()
    mc_satisfied = discriminator(img).std().item() > threshold
    if mc_satisfied: save_image(img, f'results/{num_evaluations}.png')
    return mc_satisfied


I = 0
def _adversarial_training(generator, discriminator, generator_optimiser, discriminator_optimiser, dataloader):
  global I
  for real_data in dataloader:
    # Train discriminator on real data
    discriminator_optimiser.zero_grad()
    D_real = discriminator(real_data[0])
    real_loss = F.binary_cross_entropy(D_real, torch.ones_like(D_real))  # Loss on real data
    autograd.backward(real_loss, create_graph=True)
    # R1 gradient penalty on real data
    r1_reg = 0
    for param in discriminator.parameters():
      r1_reg += param.grad.norm().mean()  
    r1_reg.backward()
    # Train discriminator on fake data
    fake_data = generator()
    D_fake = discriminator(fake_data.detach())
    fake_loss = F.binary_cross_entropy(D_fake, torch.zeros_like(D_fake))  # Loss on fake data
    fake_loss.backward()
    discriminator_optimiser.step()

    # Train generator
    generator_optimiser.zero_grad()
    D_fake = discriminator(fake_data)
    generator_loss = F.binary_cross_entropy(D_fake, torch.ones_like(D_fake))
    generator_loss.backward()
    generator_optimiser.step()
    print(real_loss.item(), fake_loss.item(), generator_loss.item())
    if I % 25 == 0: save_image(fake_data, f'results/{I}.png')
    I += 1


def evolve_seed_genomes(rand_pop, num_seeds):
  # Train a single generator and discriminator with standard GAN training + R1 grad penalty
  generator, discriminator = rand_pop[0], rand_pop[-1]  # TODO: Assumes first half of queue is generators and second half is discriminators
  generator_optimiser, discriminator_optimiser = Adam(generator.parameters(), lr=1e-4), Adam(discriminator.parameters(), lr=1e-4)
  dataset = CelebA(root='data', transform=transforms.Compose([transforms.CenterCrop(178), transforms.Resize(64), transforms.ToTensor()]), download=True)
  dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=4)
  for _ in range(10):
    _adversarial_training(generator, discriminator, generator_optimiser, discriminator_optimiser, dataloader)
  quit()
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
    children.append(child)
  return children
