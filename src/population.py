import matplotlib.pyplot as plt
from .genome import *
from .mutation import *
from .crossover import *
from typing import List
from tqdm import tqdm
import random
import copy

"""
To Do
- [ ] Smartly initialize the network weights
- [ ] While choosing random parents ensure that we don't take a single individual twice
"""

class Population:
    def __init__(self, config):
        self.config = config
        self.individuals = []
        self.genome_counter = 0

        for _ in range(self.config.population_size):
            self.individuals.append(Individual(self.new_genome(), self.config.kFitnessNotComputed))
            
    def next_genome_id(self):
        self.genome_counter += 1
        return self.genome_counter - 1

    def new_genome(self):

        genome = Genome(self.next_genome_id(), self.config.num_inputs, self.config.num_outputs)
        
        for neuron_id in range(-self.config.num_inputs, 0):
            genome.add_neuron(NeuronGene(neuron_id, "input", self.new_value()))

        for neuron_id in range(self.config.num_outputs):
            genome.add_neuron(NeuronGene(neuron_id, "output", self.new_value()))

        for input_id in range(-self.config.num_inputs, 0):
            for output_id in range(self.config.num_outputs):
                genome.add_link(LinkGene(input_id, output_id, self.new_value()))

        return genome
    
    def run(self) -> Individual:
        
        self.history = []
        
        pbar = tqdm(range(self.config.num_generations))
        for _ in pbar:
            
            info = self.config.compute_fitness(self.individuals)
            avg_score = info["avg_scores"]
            pbar.set_description(f"Avg Score: {avg_score}")
#             self.update_best()
            self.individuals = self.reproduce()
    
            self.history.append(info)
        
        return self.individuals
    
    def reproduce(self) -> List[Individual]:
        
        # sort individuals by fitness
        old_members = sorted(self.individuals, key= lambda x: x.fitness, reverse=True)
        reproduction_cutoff = int(self.config.survival_threshold * len(old_members))
        old_members = old_members[:reproduction_cutoff]
        
        new_generation = []
        spawn_size = self.config.population_size
        while spawn_size >= 0:
            
            parent1 = random.choice(old_members)
            parent2 = random.choice(old_members)

            offspring = crossover(copy.deepcopy(parent1), copy.deepcopy(parent2), self.next_genome_id())
#             self.mutate(offspring)
            new_generation.append(Individual(offspring, self.config.kFitnessNotComputed))
            
            spawn_size -= 1
        
        return new_generation
    
    def mutate(self, genome: Genome):
        
        if random.random() < self.config.prob_add_link:
            mutate_add_link(genome, self.new_value())

        if random.random() < self.config.prob_remove_link:
            mutate_remove_link(genome)
            
        if random.random() < self.config.prob_add_neuron:
            mutate_add_neuron(genome)
        
        if random.random() < self.config.prob_remove_neuron:
            mutate_remove_neuron(genome)
            
        for link in genome.links:
            if random.random() < self.config.mutation_rate:
                link.weight = self.mutate_delta(link.weight)
            
    def new_value(self):
        return self.clamp(random.gauss(self.config.init_mean, self.config.init_stdev))

    def mutate_delta(self, value):
        delta = self.clamp(random.gauss(0.0, self.config.mutate_power))
        return self.clamp(value + delta)

    def clamp(self, x):
        return min(self.config.max, max(self.config.min, x))
    
    def plot(self):
        max_scores = [info["max_scores"] for info in self.history]
        avg_scores = [info["avg_scores"] for info in self.history]
        min_scores = [info["min_scores"] for info in self.history]
        
        # Creating a figure and a set of subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plotting on each subplot
        axs[0].plot(max_scores)
        axs[0].set_title('Max Score vs Generation')

        axs[1].plot(avg_scores)
        axs[1].set_title('Avg Score vs Generation')

        # For tan(x), limiting the y-axis due to the function's asymptotic nature
        axs[2].plot(min_scores)
        axs[2].set_title('Min Score vs Generation')

        # Adjusting layout to prevent overlap
        plt.tight_layout()

        plt.show()