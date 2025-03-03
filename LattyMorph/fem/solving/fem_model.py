import torch.nn as nn

from LattyMorph.fem.solving.direct_stiffness import StiffnessMatrix
from LattyMorph.fem.solving.direct_stiffness import DirectStiffnessSolver

class FEModel(nn.Module):
    def __init__(self):
        '''
        Given a differentiable lattice, performs a finite element experiment.
        '''
        super().__init__()
        self.stiffness = StiffnessMatrix()
        self.update = DirectStiffnessSolver()

    def forward(self, lattice, experiment_setup, delta = 0):
        stiff = self.stiffness(lattice.graph, delta)
        dr, stress = self.update(lattice, stiff, experiment_setup)
        return dr, stress