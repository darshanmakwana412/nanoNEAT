

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

    def phenotype(self):
        pass