from collections import deque
from copy import deepcopy
import os
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


def evaluate_mc(generator, discriminator, threshold, num_evaluations, fixed_latent):
  with torch.no_grad():
    img = generator(fixed_latent)
    mc_satisfied = torch.sigmoid(discriminator(img)).std().item() > threshold
    save_image(img, f'results/{num_evaluations}_satisfied.png' if mc_satisfied else f'results/{num_evaluations}.png')
    return mc_satisfied


def _adversarial_training(generator, discriminator, generator_optimiser, discriminator_optimiser, dataloader, latent_size, epoch, batch_size, fixed_latent, device):
  label_ones, label_zeros = torch.ones(batch_size, device=device), torch.zeros(batch_size, device=device)
  for i, real_data in enumerate(dataloader):
    # Train discriminator
    discriminator_optimiser.zero_grad()
    fake_data = generator(torch.randn(real_data[0].size(0), latent_size, device=device))
    D_real, D_fake = discriminator(real_data[0].to(device=device)), discriminator(fake_data.detach())
    discriminator_loss = F.binary_cross_entropy_with_logits(D_real, label_ones) + F.binary_cross_entropy_with_logits(D_fake, label_zeros)
    discriminator_loss.backward()
    discriminator_optimiser.step()

    # Train generator
    for j in range(2):
      generator_optimiser.zero_grad()
      D_fake = discriminator(fake_data if j == 0 else generator(torch.randn(real_data[0].size(0), latent_size, device=device)))
      generator_loss = F.binary_cross_entropy_with_logits(D_fake, label_ones)
      generator_loss.backward()
      generator_optimiser.step()

    if i % 500 == 0:
      print(epoch, i, discriminator_loss.item(), generator_loss.item())
      with torch.no_grad():
        generator.eval()
        save_image(generator(fixed_latent), f'results/adversarial_{epoch}_{i}.png')
        generator.train()


def evolve_seed_genomes(rand_pop, num_seeds, latent_size, learning_rate, batch_size, epochs, fixed_latent, mutation_rate, device):
  generator, discriminator = rand_pop[0], rand_pop[-1]  # TODO: Assumes first half of queue is generators and second half is discriminators
  generator.to(device=device)
  discriminator.to(device=device)

  if os.path.exists('models/generator.pt') and os.path.exists('models/discriminator.pt'):
    # Load pretrained models
    generator.load_state_dict(torch.load('models/generator.pt', map_location='cpu'))
    discriminator.load_state_dict(torch.load('models/discriminator.pt', map_location='cpu'))
  else:
    # Train a single generator and discriminator with standard GAN training
    dataset = CelebA(root='data', transform=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(148), transforms.Resize([64])]), download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=6)
    generator_optimiser, discriminator_optimiser = Adam(generator.parameters(), lr=learning_rate), Adam(discriminator.parameters(), lr=learning_rate)

    for epoch in range(epochs):
      _adversarial_training(generator, discriminator, generator_optimiser, discriminator_optimiser, dataloader, latent_size, epoch, batch_size, fixed_latent, device)
      torch.save(generator.state_dict(), f'models/generator_{epoch}.pt')
      torch.save(discriminator.state_dict(), f'models/discriminator_{epoch}.pt')
  
  # Create viable population from trained generator and discriminator
  generator.eval()
  discriminator.eval()
  rand_pop = [reproduce([generator], mutation_rate)[0] for _ in range(num_seeds)] + [reproduce([discriminator], mutation_rate)[0] for _ in range(num_seeds)]
  return rand_pop


def remove_oldest(viable_pop, num_removals):
  youngest = np.argsort([s.age for s in viable_pop])
  return deque([viable_pop[i] for i in youngest[:len(viable_pop) - num_removals]])


def reproduce(parents, mutation_rate):
  children = []
  for parent in parents:
    if isinstance(parent, Discriminator): parent.remove_spectral_norm()  # Remove spectral norm to allow deepcopy
    child = deepcopy(parent)
    child.age = 0
    if isinstance(child, Discriminator): child.usage = 0
    for parameter in child.parameters():
      parameter.data.add_(mutation_rate * torch.randn_like(parameter.data))
    if isinstance(parent, Discriminator):
      parent.add_spectral_norm()  # Add back spectral norm
      child.add_spectral_norm()
    children.append(child)
  return children
