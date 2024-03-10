import matplotlib.pyplot as plt
from .genome import *
from .mutation import *
from .crossover import *
from .graph import *
from typing import List
from tqdm import tqdm
import random
import copy
import os

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
        fig, ax = plt.subplots(figsize=(5, 5))
        
        pbar = tqdm(range(self.config.num_generations))
        for generation_id in pbar:
            
            info = self.config.compute_fitness(self.individuals, generation_id, ax)
            
            avg_score = info["avg_scores"]
            scores = info["scores"]
            pbar.set_description(f"Avg Score: {avg_score}")
            
            idx, max_score = max(enumerate(scores), key=lambda x: x[1])
            best_individual = self.individuals[idx]

            self.individuals = self.reproduce()
    
            self.history.append(info)
        
        if self.config.plot_info:
            self.plot_info()
        
        return best_individual
    
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
            offspring.parent1_genome = parent1.genome
            offspring.parent2_genome = parent2.genome
#             self.mutate(offspring)
#             print(parent1.genome)
#             print(parent2.genome)
            network = Graph(offspring)

            if not network._is_cyclic():

                new_generation.append(Individual(offspring, self.config.kFitnessNotComputed))

                spawn_size -= 1
        
        return new_generation
    
    def mutate(self, genome: Genome) -> None:
        
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
    
    def plot_info(self):
        
#         max_scores = [info["max_scores"] for info in self.history]
        avg_scores = [info["avg_scores"] for info in self.history]
#         min_scores = [info["min_scores"] for info in self.history]
        generations = [i for i in range(1, len(avg_scores) + 1)]

        # Create the directory if it does not exist
        output_dir = os.path.join(self.config.output_dir, self.config.exp_name)
        os.makedirs(output_dir, exist_ok=True)

        # Save the plot to the specified path
        plt.figure(figsize=(10, 6))
#         plt.plot(generations, max_scores, label='Max Score', marker='o')
        plt.plot(generations, avg_scores, label='Reward', marker='s')
#         plt.plot(generations, min_scores, label='Min Score', marker='^')

        plt.title('Reward at inference')
        plt.xlabel('iter')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "info.png")
        plt.savefig(plot_path)