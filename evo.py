from collections import deque
from copy import deepcopy
import numpy as np
import torch
from torch import autograd, jit, nn
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


def _adversarial_training(generator, discriminator, generator_optimiser, discriminator_optimiser, dataloader, latent_size, epoch, device):
  for i, real_data in enumerate(dataloader):
    print(i)
    # Train discriminator
    discriminator_optimiser.zero_grad()
    fake_data = generator(torch.randn(real_data[0].size(0), latent_size, device=device))
    D_real, D_fake = discriminator(real_data[0].to(device=device)), discriminator(fake_data.detach())
    discriminator_loss = F.binary_cross_entropy_with_logits(D_real, torch.ones_like(D_real)) + F.binary_cross_entropy_with_logits(D_fake, torch.zeros_like(D_fake))
    discriminator_loss.backward()
    discriminator_optimiser.step()

    # Train generator
    generator_optimiser.zero_grad()
    D_fake = discriminator(fake_data)
    generator_loss = F.binary_cross_entropy_with_logits(D_fake, torch.ones_like(D_fake))
    generator_loss.backward()
    generator_optimiser.step()
    if i % 500 == 0:
      print(epoch, i, discriminator_loss.item(), generator_loss.item())
      save_image(fake_data, f'results/{epoch}_{i}.png')


def evolve_seed_genomes(rand_pop, num_seeds, latent_size, learning_rate, batch_size, device):
  # Train a single generator and discriminator with standard GAN training + R1 grad penalty
  generator, discriminator = rand_pop[0], rand_pop[-1]  # TODO: Assumes first half of queue is generators and second half is discriminators
  generator.to(device=device)
  discriminator.to(device=device)
  generator_optimiser, discriminator_optimiser = Adam(generator.parameters(), lr=learning_rate), Adam(discriminator.parameters(), lr=learning_rate)

  dataset = CelebA(root='data', transform=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(148), transforms.Resize([64])]), download=True)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=6)

  epoch = 0
  while True:
    _adversarial_training(generator, discriminator, generator_optimiser, discriminator_optimiser, dataloader, latent_size, epoch, device)
    epoch += 1
    torch.save(generator.state_dict(), f'models/generator_{epoch}.pth')
    torch.save(discriminator.state_dict(), f'models/discriminator_{epoch}.pth')
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
