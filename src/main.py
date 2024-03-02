from typing import Dict
import numpy as np
from tqdm import tqdm
from .population import Population

class Config:
    def __init__(self):
        
        # simulation params
        self.population_size = 500
        self.num_generations = 100
        self.survival_threshold = 0.2
        self.max_episode_length = 10000
        
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
        self.mutation_rate = 0.1
        self.mutate_power = 1.2
        self.replace_rate = 0.05
        
    def compute_fitness(self, individuals) -> Dict:
        
        scores = []
        lens = []
        
        for individual in individuals:
            
            env = SnakeEnv()
            obs = env.reset()
            done = False
            
            graph = Graph(individual.genome)
            score = 0
    
            for i in range(self.max_episode_length):
            
                outputs = graph.forward(obs)
                action = np.argmax(outputs)
                obs, reward, done, info = env.step(action)
                score += reward
                
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
pool.plot()