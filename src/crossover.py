import random
from .genome import *

def crossover_neuron(gene1: NeuronGene, gene2: NeuronGene) -> NeuronGene:

    # Ensure the crossover occurs between neurons of the same type from two individuals
    assert(gene1.neuron_id == gene2.neuron_id)
    assert(gene1.type == gene2.type)

    # Randomly choose the bias and activations from either of the parents
    neuron_id = gene1.neuron_id
    bias = random.choice([gene1.bias, gene2.bias])
    activation = random.choice([gene1.activation, gene2.activation])
    return NeuronGene(neuron_id, gene1.type, bias, activation)

def crossover_link(gene1: LinkGene, gene2: LinkGene) -> LinkGene:
    # Ensure the crossover occurs between neurons of the same type from two individuals
    assert(gene1.input_id == gene2.input_id)
    assert(gene1.output_id == gene2.output_id)
    input_id = gene1.input_id
    
    # Randomly choose the weights from either of the parents
    output_id = gene2.output_id
    weight = random.choice([gene1.weight, gene2.weight])
    is_enabled = random.choice([gene1.is_enabled, gene2.is_enabled])
    return LinkGene(input_id, output_id, weight, is_enabled)

def crossover(dominant: Individual, recessive: Individual, genome_id: int) -> Genome:
    
    offspring = Genome(genome_id, dominant.genome.num_inputs, dominant.genome.num_outputs)
    
    # Inherit neuron gene
    for dominant_neuron in dominant.genome.neurons:
        
        neuron_id = dominant_neuron.neuron_id
        recessive_neuron = next((neuron for neuron in recessive.genome.neurons if neuron.neuron_id == neuron_id), None)
        
        if recessive_neuron:
            offspring.neurons.append(crossover_neuron(dominant_neuron, recessive_neuron))
        else:
            offspring.neurons.append(dominant_neuron)
            
    # Inherit link gene
    for dominant_link in dominant.genome.links:
        
        input_id = dominant_link.input_id
        output_id = dominant_link.output_id
        recessive_link = next((link for link in recessive.genome.links if ((link.input_id == input_id) and (link.output_id == output_id))), None)
        
        if recessive_link:
            offspring.links.append(crossover_link(dominant_link, recessive_link))
        else:
            offspring.links.append(dominant_link)
            
    return offspring