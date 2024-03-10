from src.population import Population
from envs.snake import SnakeEnv
from src.graph import Graph

from typing import Dict
from tqdm import tqdm
import numpy as np
import os

class Config:
    def __init__(self):
        
        # General config
        self.exp_name = "snake-v0.4"
        self.plot_info = True
        self.output_dir = "./outputs/"
        self.animate = False
        
        # simulation params
        self.population_size = 25
        self.num_generations = 50
        self.survival_threshold = 0.1
        self.max_episode_length = 100000
        
        # genome params
        self.num_inputs = 4
        self.num_outputs = 4
        self.kFitnessNotComputed = 0
        
        # structural mutation params
        self.prob_add_link = 0.5
        self.prob_remove_link = 0.5
        self.prob_add_neuron = 0.5
        self.prob_remove_neuron = 0.5

        # non structural mutation params
        self.init_mean = 0.0
        self.init_stdev = 1.0
        self.min = -20.0
        self.max = 20.0
        self.mutation_rate = 0.5
        self.mutate_power = 1.2
        self.replace_rate = 0.05
        
    def compute_fitness(self, individuals, generation: int, ax) -> Dict:
        
        scores = []
        lens = []
        
        for individual_id, individual in enumerate(individuals):
#             print(f"computing fitness for individual {individual_id}")
#             print(f"parent 1: {individual.parent1_genome}")
#             print(f"parent 1: {individual.parent2_genome}")
            
            if self.animate:
                save_dir = os.path.join(self.output_dir, f"{self.exp_name}/generation_{generation}/individual_{individual_id}")
                os.makedirs(save_dir, exist_ok=True)
                
            env = SnakeEnv()
            obs = env.reset()
            done = False
            
            graph = Graph(individual.genome)
            score = 0
    
            for i in range(self.max_episode_length):
                try:
                    outputs = graph.forward(obs)
                except:
                    print(f"Skipping individual: {individual_id}")
                    scrore = -1
                    break
                action = np.argmax(outputs)
                obs, reward, done, info = env.step(action)
                score += reward
                
                if self.animate:
                    render_path = os.path.join(save_dir, f"frame_{i}.png")
                    env.render_save(render_path, ax)
                
                if done:
                    break
                    
            individual.fitness = score
            scores.append(score)
            lens.append(i)
            
        return {
            "scores": scores,
            "episode_lengths": lens,
            "avg_scores": sum(scores)/len(scores),
            "max_scores": max(scores),
            "min_scores": min(scores)
        }

pool = Population(Config())
winner = pool.run()
print(winner.genome)