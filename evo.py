from collections import deque
from copy import deepcopy
import os
from miditime.miditime import MIDITime
import numpy as np
import torch
from torchvision.utils import save_image

from models import Discriminator


I = 0


def visualise_songs(songs):
  canvas = torch.ones(songs.size(0), 1, 100, 25)  # Pitch x timesteps
  for n, song in enumerate(songs):
    for t, note in enumerate(song):
      _, pitch, velocity, duration = note
      canvas[n, :, pitch, t:t + duration] = 1 - velocity.item() / 200
  return canvas


def save_song(songs, song_plots):
  global I
  for n, song in enumerate(songs):
    os.makedirs(f'results/{I}', exist_ok=True)
    midi = MIDITime(130, f'results/{I}/{n}.mid')  # Save file at 130 BPM
    midi.add_track(song.numpy())
    midi.save_midi()
  save_image(song_plots, f'results/{I}/plots.png')


def evaluate_mc(generator, discriminator):
  global I
  I += 1
  timesteps = 20
  with torch.no_grad():
    songs = generator()
    songs = torch.cat([torch.linspace(0, timesteps - 1, timesteps, dtype=torch.int64).view(1, timesteps, 1).expand(songs.size(0), songs.size(1), 1), songs], dim=2)  # Add time
    song_plots = visualise_songs(songs)
    mc_satisfied = discriminator(song_plots).std().item() > 0.3
    if mc_satisfied: save_song(songs, song_plots)
    return mc_satisfied


# TODO: Implement
def evolve_seed_genomes(rand_pop, num_seeds):
  return rand_pop


def remove_oldest(viable_pop, num_removals):
  youngest = np.argsort([s.age for s in viable_pop])
  return deque([viable_pop[i] for i in youngest[:len(viable_pop) - num_removals]])


def reproduce(parents):
  children = []
  for parent in parents:
    child = deepcopy(parent)
    child.age = 0
    if isinstance(child, Discriminator): child.usage = 0
    for parameter in child.parameters():
      parameter.data.add_(0.2 * torch.randn_like(parameter.data))
    children.append(child)
  return children
