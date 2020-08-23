from collections import deque
import torch
from torchvision.utils import save_image

from evo import evaluate_mc, evolve_seed_genomes, reproduce
from models import Discriminator, generate_random_population, Generator


batch_size = 32  # Number of individuals to evaluate simultaneously
num_seeds = 5  # Number of seed genomes to evolve that satisfy the MC
resource_limit = 2  # Max number of evaluations that count towards MC
viable_pop_capacity = 50

# Evolve seed genomes that satisfy MC
rand_pop = generate_random_population(20)
viable_pop = deque(evolve_seed_genomes(rand_pop, num_seeds))

for _ in range(100):
  # Reproduce children and add parents back into queue
  parents = [viable_pop.popleft() for _ in range(batch_size)]
  children = reproduce(parents)
  [viable_pop.append(p) for p in parents]

  for i, child in enumerate(children):  # TODO: Somehow make sure there are both Gs and Ds?
    if isinstance(child, Generator):
      for _child in children:
        if isinstance(_child, Discriminator) and _child.usage < resource_limit:
          eval_indiv = _child
          break
    else:
      for j in range(i, len(children)):
        if isinstance(children[j], Generator):
          eval_indiv = children[j]
        break
    if eval_indiv is None: break

    mc_satisfied = evaluate_mc(child, eval_indiv) if isinstance(child, Generator) else evaluate_mc(eval_indiv, child)
    if mc_satisfied: viable_pop.append(child)
    eval_indiv = None

  if len(viable_pop) > viable_pop_capacity:
    num_removals = len(viable_pop) - viable_pop_capacity
    # TODO: remove_oldest(viable_pop, num_removals)
