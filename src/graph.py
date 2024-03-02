from collections import deque
from .genome import *
import numpy as np
import random
import math

class Node:
    def __init__(self, node_id: int, ntype: str, bias: float, activation: str) -> None:
        self.node_id = node_id
        self.type = ntype
        self.bias = bias
        self.activation = activation
        self.inputs = []
        self.outputs = []
        self.value = None
        
    def add_input(self, input_id, weight) -> None:
        self.inputs.append((input_id, weight))
        
    def add_output(self, output_id, weight) -> None:
        self.outputs.append((output_id, weight))

class Graph:
    def __init__(self, genome: Genome):
        
        self.genome = genome
        self.nodes = dict()
        
        for n in genome.neurons:
            self.nodes[n.neuron_id] = Node(n.neuron_id, n.type, n.bias, n.activation)
            
        for link in genome.links:
            input_id = link.input_id
            output_id = link.output_id
            weight = link.weight
            
            if link.is_enabled:
                self.nodes[input_id].add_output(output_id, weight)
                self.nodes[output_id].add_input(input_id, weight)
            
    def forward(self, inputs: List[float], verbose: bool = False) -> List[float]:
        
        input_ids = self.genome.make_input_ids()
        output_ids = self.genome.make_output_ids()
        neurons = self.genome.neurons

        neuron_queue = deque(input_ids)
        
        assert (len(inputs) == len(input_ids)), f"Size of input should match the size of input neurons in genome, Size of input: {len(inputs)}, Size of input neurons: {len(input_ids)}"
        
        for value, input_id in zip(inputs, input_ids):
            self.nodes[input_id].value = value
        
        if verbose:
            num_iter = 0
            max_len = 0
        while len(neuron_queue):
            
            nid = neuron_queue.popleft()
            node = self.nodes[nid]
            
            status = self.node_pass(node)
            
            for (output_id, weight) in node.outputs:
                if self.nodes[output_id].value == None:
                    neuron_queue.append(output_id)
                    
            if not status:
                neuron_queue.append(nid)
            
            if verbose:
                max_len = max(max_len, len(neuron_queue))
                num_iter += 1
            
        outputs = []
        for output_id in output_ids:
            value = self.nodes[output_id].value
            assert(value != None)
            outputs.append(value)
        
        if verbose:
            print(f"Num iterations: {num_iter}")
            print(f"Max Queue len: {max_len}")
            
        return outputs
            
    def node_pass(self, node):
        
        if node.type != "input" and node.value == None:
            
            value = 0
            for (input_id, weight) in node.inputs:
                input_value = self.nodes[input_id].value
                if input_value == None:
                    return False
                value += input_value * weight
            value += node.bias
            value = self.activate(value, node.activation)
            node.value = value
            
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