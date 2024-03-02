import random
from .genome import *

"""
There are 4 types of structural mutations that can occur in the genome
- Add a link between two neuron (`mutate_add_link`)
- Remove a link between two neurons (`mutate_remove_link`)
- Add a new hidden neuron (`mutate_add_neuron`)
- Remove a hidden neuron (`mutate_remove_neuron`)

Non structural mutations occur by randomly changing properties of the link or neuron

To Do:
- [*] create a function `would_create_cycle` for checking cycles
    No need for cycle checking as the new implementation uses an iterative graph traversal
- [*] create a link mutator object `link_mutator` for generating new weights smartly
    Creating new weights using the `new_value` function of `population` class
- [*] create a neuron mutator object `neuron_mutator` for generating new neurons smartly
"""

"""
    Structural Mutations
"""

def mutate_add_link(genome: Genome, weight: float) -> None:
    """
        Adds a new link to the genome (or enables an existing one), ensuring no cycles

        Args:
        genome: The genome to mutate.
    """

    input_id = random.choice([n.neuron_id for n in genome.neurons if n.type != "output"])
    output_id = random.choice([n.neuron_id for n in genome.neurons if n.type != "input"])

    # Check for existing link
    for link in genome.links:
        if link.input_id == input_id and link.output_id == output_id:
            link.is_enabled = True  # Enable if found
            return

    # Assuming you have a 'link_mutator' object for generating new weights
#     weight = link_mutator.new_value(input_id, output_id)
    new_link = LinkGene(input_id, output_id, weight, True)  # Enabled by default
    genome.links.append(new_link)

def mutate_remove_link(genome: Genome):
    """
        Removes a randomly choosen link from the genome

        Args:
        genome: The genome to mutate.
    """

    # Check if there are any links to remove
    if len(genome.links) == 0:
        return
    
    link = random.choice(genome.links)
    genome.links.remove(link)
    
def mutate_add_neuron(genome: Genome, bias: float = 0, activation: str = "relu"):
    """
        Adds a new neuron to the genome by splitting a random link.

        Args:
        genome: The genome to mutate.
    """

    # Check if there are any links to split
    if len(genome.links) == 0:
        return

    # Choose a random link to split
    link_to_split = random.choice(genome.links)
    link_to_split.is_enabled = False  # Disable the chosen link

    # Create a new neuron
#     new_neuron = neuron_mutator.new_neuron()
    new_neuron = NeuronGene(len(genome.neurons), "hidden", bias, activation)  # Assuming bias and activation logic
    genome.neurons.append(new_neuron)

    # Create new links from the split point to the new neuron and the original output
    genome.links.append(LinkGene(link_to_split.input_id, new_neuron.neuron_id, 1, True))
    genome.links.append(LinkGene(new_neuron.neuron_id, link_to_split.output_id, link_to_split.weight, True))
    
def mutate_remove_neuron(genome: Genome):
    """
        Removes a randomly chosen hidden neuron from the genome and removes associated links.

        Args:
        genome: The genome to mutate.
    """
    
    hidden_neurons = [n for n in genome.neurons if n.type == "hidden"]
    if len(hidden_neurons) == 0:
        return

    # Choose a random hidden neuron
    neuron_to_remove = random.choice(hidden_neurons)

    # Remove associated links
    links_to_remove = [
        link for link in genome.links if ((link.input_id == neuron_to_remove.neuron_id) and (link.output_id == neuron_to_remove.neuron_id))
    ]
    for link in links_to_remove:
        genome.links.remove(link)

    # Remove the neuron itself
    genome.neurons.remove(neuron_to_remove)