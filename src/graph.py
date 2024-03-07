from collections import deque
from .genome import *
import numpy as np
import random
import math

"""
To Do:
- [] Add support for cycle checking inside a genotype
"""

class Node:
    def __init__(self, node_id: int, ntype: str, bias: float, activation: str) -> None:
        self.node_id = node_id
        self.type = ntype
        self.bias = bias
        self.activation = activation
        self.inputs = []
        self.outputs = []
        
    def add_input(self, input_id: int, weight: float) -> None:
        self.inputs.append((input_id, weight))
        
    def add_output(self, output_id: int, weight: float) -> None:
        self.outputs.append((output_id, weight))

class Graph:
    def __init__(self, genome: Genome):
        
        self.genome = genome
        self.nodes = dict()
#         print(genome)
#         print(f"Printing genome: {genome.genome_id}")
        
        for n in genome.neurons:
#             print(n.neuron_id)
            self.nodes[n.neuron_id] = Node(n.neuron_id, n.type, n.bias, n.activation)
            
        for link in genome.links:
            input_id = link.input_id
            output_id = link.output_id
            weight = link.weight
            
            if link.is_enabled:
#                 print(input_id, output_id, weight)
                self.nodes[input_id].add_output(output_id, weight)
                self.nodes[output_id].add_input(input_id, weight)
                
    def _is_cyclic(self) -> bool:
        
        neuron_ids = [n.neuron_id for n in self.genome.neurons]

        for nid in neuron_ids:
            self.visited = {}
            if nid not in self.visited:
                if self.dfs(nid):
                    return True
            
        return False
    
    def dfs(self, nid: int) -> bool:
        node = self.nodes[nid]
        self.visited[nid] = True
        for (output_id, weight) in node.outputs:
            if output_id not in self.visited:
                self.dfs(output_id)
            else:
                return True
        return False
            
    def forward(self, inputs: List[float], verbose: bool = False) -> List[float]:
        
        input_ids = self.genome.make_input_ids()
        output_ids = self.genome.make_output_ids()
        neurons = self.genome.neurons
        self.values = {}
        
        assert(len(inputs) == len(input_ids)), f"Size of input should match the size of input neurons in genome, Size of input: {len(inputs)}, Size of input neurons: {len(input_ids)}"
        
        for value, input_id in zip(inputs, input_ids):
            self.values[input_id] = value

        num_iter = 0
        max_len = 0
        neuron_queue = deque(input_ids)
        while len(neuron_queue):
            nid = neuron_queue.popleft()
            node = self.nodes[nid]
            
            status = self.node_pass(node)
            
            for (output_id, weight) in node.outputs:
                if output_id not in self.values:
                    neuron_queue.append(output_id)
                    
            if not status:
                neuron_queue.append(nid)
            
            max_len = max(max_len, len(neuron_queue))
            num_iter += 1
                
            if num_iter > 100000:
                print("Max length of the queue excedded")
                break
            
        outputs = []
        for output_id in output_ids:
            if output_id not in self.values:
                print(self.values)
                print(self.genome)
                print("illegal forward pass")
                print(output_ids)
            value = self.values[output_id]
            outputs.append(value)
        
        if verbose:
            print(f"Num iterations: {num_iter}")
            print(f"Max Queue len: {max_len}")
            
        return outputs
            
    def node_pass(self, node):
        # check if the neuron has already been computed or is an input neuron
        if node.type != "input" and node.node_id not in self.values:
            
            value = 0
            for (input_id, weight) in node.inputs:
                if input_id not in self.values:
                    return False
                input_value = self.values[input_id]
                value += input_value * weight
            value += node.bias
            value = self.activate(value, node.activation)
            self.values[node.node_id] = value
            
        return True
                    
    def activate(self, x: int, activation: str, alpha: float = 0.01) -> float:
        
        if activation == "sigmoid":
            return 1 / (1 + math.exp(-x))
        elif activation == "tanh":
            return math.tanh(x)
        elif activation == "relu":
            return max(x, 0)
        elif activation == "leaky_relu":
            return max(alpha * x, x)
        else:
            raise Exception("Undefined activation function")