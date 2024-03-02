from typing import List

class NeuronGene:
    """
    Base class for representing gene of a single neuron
    """
    def __init__(self, neuron_id: int, ntype: str = "hidden", bias: float = 0, activation: str = "relu") -> None:
        self.neuron_id = neuron_id
        self.type = ntype # input | output | hidden
        self.bias = bias
        self.bias = 0
        self.activation = activation
        
    def __str__(self) -> str:
        return f"NeuronGene: {self.neuron_id}\n    Type: {self.type}\n    Bias: {self.bias}\n    Activation: {self.activation}"
        
class LinkGene:
    def __init__(self, input_id: int, output_id: int, weight: float = 0, is_enabled: bool = True) -> None:
        self.input_id = input_id
        self.output_id = output_id
        self.weight = weight
        self.is_enabled = is_enabled
        
    def __str__(self) -> str:
        return f"NeuronGene:\n    input_id: {self.input_id}\n    output_id: {self.output_id}\n    weight: {self.weight}\n    is_enabled: {self.is_enabled}"
        
class Genome:
    def __init__(self, genome_id: int, num_inputs: int, num_outputs: int) -> None:
        self.genome_id = genome_id
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        self.neurons = []
        self.links = []
        
    def add_neuron(self, neuron: NeuronGene) -> None:
        self.neurons.append(neuron)
        
    def add_link(self, link: LinkGene) -> None:
        self.links.append(link)
        
    def make_input_ids(self) -> List[int]:
        return [n.neuron_id for n in self.neurons if n.type == "input"]
    
    def make_output_ids(self) -> List[int]:
        return [n.neuron_id for n in self.neurons if n.type == "output"]
    
    def __str__(self) -> str:
        input_neurons = len([n.neuron_id for n in self.neurons if n.type == "input"])
        output_neurons = len([n.neuron_id for n in self.neurons if n.type == "output"])
        hidden_neurons = len([n.neuron_id for n in self.neurons if n.type == "hidden"])
        num_links = len(self.links)
        return f"Genome: {self.genome_id}\n    input neurons: {input_neurons}\n    output neurons: {output_neurons}\n    hidden neurons: {hidden_neurons}\n    num links: {num_links}"
        
class Individual:
    """
    Base class of an Individual in a population
    """
    def __init__(self, genome: Genome, fitness: float) -> None:
        self.genome = genome
        self.fitness = fitness