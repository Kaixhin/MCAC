import argparse
from collections import deque
import os
import torch
from torchvision.utils import save_image

from evo import evaluate_mc, evolve_seed_genomes, remove_oldest, reproduce
from models import Discriminator, generate_random_population, Generator


# Setup
os.makedirs(f'models', exist_ok=True)  # Make models directory
os.makedirs(f'results', exist_ok=True)  # Make results directory
parser = argparse.ArgumentParser(description='MCAC')
parser.add_argument('--seed', type=int, default=1, metavar='SEED', help='Random seed')
parser.add_argument('--generator', type=str, default='StyleGAN', choices=['DCGAN', 'StyleGAN', 'CPPN'], metavar='GENERATOR', help='Generator type')
parser.add_argument('--initial-pop', type=int, default=50, metavar='INITIAL', help='Initial population size')
parser.add_argument('--num-seeds', type=int, default=5, metavar='SEEDS', help='Number of seed genomes to evolve that satisfy the MC')
parser.add_argument('--viable-pop-capacity', type=int, default=200, metavar='CAPACITY', help='Viable population capacity')
parser.add_argument('--max-epochs', type=int, default=500, metavar='EPOCHS', help='Max number of epochs')
parser.add_argument('--evaluation-size', type=int, default=64, metavar='SIZE', help='Number of individuals to evaluate simultaneously')
parser.add_argument('--resource-limit', type=int, default=5, metavar='LIMIT', help='Max number of evaluations that count towards MC')
parser.add_argument('--mutation-rate', type=float, default=0.2, metavar='MUTATION', help='Mutation rate')
parser.add_argument('--criterion-threshold', type=float, default=0.3, metavar='THRESHOLD', help='Criterion threshold')
parser.add_argument('--latent-size', type=int, default=128, metavar='LATENT', help='Latent size')
parser.add_argument('--hidden-size', type=int, default=64, metavar='HIDDEN', help='Hidden size')
parser.add_argument('--learning-rate', type=float, default=1e-4, metavar='RATE', help='Learning rate')
parser.add_argument('--batch-size', type=int, default=64, metavar='SIZE', help='Adversarial training batch size')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


if __name__ == '__main__':
  # Evolve seed genomes that satisfy MC
  rand_pop = generate_random_population(args.generator, args.initial_pop, args.latent_size, args.hidden_size)
  viable_pop = deque(evolve_seed_genomes(rand_pop, args.num_seeds, args.latent_size, args.learning_rate, args.batch_size, device))


  num_evaluations = 0
  for _ in range(args.max_epochs):
    # Increase age of all solutions in viable population
    for s in viable_pop:
      s.age += 1

    # Reproduce children and add parents back into queue
    parents = [viable_pop.popleft() for _ in range(args.evaluation_size)]
    children = reproduce(parents, args.mutation_rate)
    [viable_pop.append(p) for p in parents]

    for i, child in enumerate(children):  # TODO: Somehow make sure there are both generators and discriminators?
      if isinstance(child, Generator):
        for _child in children:
          if isinstance(_child, Discriminator) and _child.usage < args.resource_limit:
            generator = child
            eval_indiv = discriminator = _child
            break
      else:
        for j in range(i, len(children)):
          if isinstance(children[j], Generator):
            discriminator = child
            eval_indiv = generator = children[j]
          break
      if eval_indiv is None: break

      num_evaluations += 1
      mc_satisfied = evaluate_mc(generator, discriminator, args.criterion_threshold, num_evaluations)
      if mc_satisfied:
        discriminator.usage += 1
        viable_pop.append(child)
      eval_indiv = None

    if len(viable_pop) > args.viable_pop_capacity:
      num_removals = len(viable_pop) - args.viable_pop_capacity
      viable_pop = remove_oldest(viable_pop, num_removals)
