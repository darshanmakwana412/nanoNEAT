import matplotlib.pyplot as plt
from utils import plot_edge, plot_square

class Genome:
    """Base class for the Genome"""

    def __init__(self) -> None:
        
        """
        node genes
        0 -> input node
        1 -> hidden node
        2 -> output node
        """
        self.node_genes = [0, 0, 0, 2, 1]
        
        """
        Input node
        Output node
        Weight of the connection
        Enabled/1
        Ennovation
        """
        self.connect_genes = [
            (0, 3, 0.7, 1, 1),
            (1, 3, -0.5, 0, 2),
            (2, 3, 0.5, 1, 3),
            (1, 4, 0.2, 1, 4),
            (4, 3, 0.4, 1, 5),
            (0, 4, 0.6, 1, 6),
            (3, 4, 0.6, 1, 11),
        ]

    def plot(self) -> None:

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-1, 3)
        ax.set_ylim(-1, 4)

        # Node positions by layer
        layers = {0: [], 1: [], 2: []}
        for i, gene in enumerate(self.node_genes):
            if gene == 0:  # Input
                layers[0].append(i)
            elif gene == 1:  # Hidden
                layers[1].append(i)
            else:  # Output
                layers[2].append(i)

        # Calculate positions
        positions = {}
        for layer, nodes in layers.items():
            y_step = 4 / (len(nodes) + 1)
            for i, node in enumerate(nodes):
                positions[node] = (layer, (i+1) * y_step)

        # Draw nodes with enhanced aesthetics
        for node, pos in positions.items():
            is_hidden = self.node_genes[node] == 1
            plot_square(ax, pos, node, is_hidden=is_hidden)

        # Draw connections with enhanced aesthetics
        for src, dst, weight, enabled, _ in self.connect_genes:
            start_pos = positions[src]
            end_pos = positions[dst]
            plot_edge(ax, start_pos, end_pos, weight, enabled)

        plt.axis('off')
        plt.savefig("plot.png")

g1 = Genome()
g1.plot()