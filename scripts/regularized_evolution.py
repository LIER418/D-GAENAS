

import queue
import random
import time
import uuid

from absl import logging


class Individual:

  def __init__(self, gene, fitness):
    self.gene = gene
    self.fitness = fitness

  def __str__(self):
    return f'gene: {self.gene}\nfitness: {self.fitness}'

  def __eq__(self, other):
    return isinstance(other, Individual) and self.gene == other.gene

  def serialize_gene(self):
    return f'{self.gene}'


class Population:

  def __init__(
      self,
      population_size,
      tournament_size,
      mutation_probability=0.9,
      max_mutations=-1,
      history_writer=None,
      other_config=None):
    self._population_size = population_size
    self._tournament_size = tournament_size
    self._mutation_probability = mutation_probability
    self._history_writer = history_writer
    self._max_mutations = max_mutations
    self._history_counter = 0
    self._queue = queue.Queue()
    self._individuals = {}
    self._start_time = time.time()
    self._cfg = other_config
    self.create_initial_population()

  def create_initial_population(self):
    pass

  def _sample_tournament(self):
    return random.sample(
        list(self._individuals.items()),
        k=min(len(self), self._tournament_size))

  def get_parent(self):
    probability = random.uniform(0, 1)
    while probability > self._mutation_probability:
      individual = min(
          self._sample_tournament(), key=lambda x: x[1].fitness)[1]
      self.add_to_population(individual.gene, individual.fitness)
      probability = random.uniform(0, 1)

    individual = min(
        self._sample_tournament(), key=lambda x: x[1].fitness)[1]
    return individual.gene

  def add_to_population(self, gene, fitness, **kwargs):
    individual_id = str(uuid.uuid4())
    individual = Individual(gene=gene, fitness=fitness)
    self._individuals[individual_id] = individual
    self._queue.put_nowait(individual_id)
    if self._history_writer is not None:
      self._history_writer.write({
          'received_time': str(time.time()),
          'fitness': individual.fitness,
          'gene': individual.serialize_gene(),
          **kwargs,
      })
      if time.time() - self._start_time > 300:
        logging.info('Flush the writer and wait the data server complete.')
        self._history_writer.wait()
        self._start_time = time.time()

    logging.info('Put %s into population with id %s', individual, individual_id)
    if len(self) > self._population_size:
      pop_individual_id = self._queue.get_nowait()
      pop_individual = self._individuals.pop(pop_individual_id)
      logging.info(
          'Remove %s from population with id %s',
          pop_individual, pop_individual_id)

    self._history_counter += 1
    if self._max_mutations > 0 and self._history_counter >= self._max_mutations:
      if self._history_writer is not None:
        self._history_writer.wait()
      return False
    else:
      return True

  def __len__(self):
    return len(self._individuals)
