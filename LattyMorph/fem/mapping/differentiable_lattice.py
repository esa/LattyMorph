import torch
import numpy as np
import dgl
import dgl.function as fn
from collections import defaultdict

from LattyMorph.fem.mapping import Constraints
from LattyMorph.fem.mapping import ExternalDisplacement

from LattyMorph.fem.tools.utils import get_BeamShapeFactor
from LattyMorph.draw import draw_graph

class DifferentiableLattice():
    def __init__(self, lattice_setup):
        '''
        Differentiable lattice object.
        Used together with FEModel to enable differentiable finite element simulations.
        '''
        self.graph = self.dict_to_dgl(lattice_setup, option="no plot")
        self.graph_plot = self.dict_to_dgl(lattice_setup, option="yes plot")
        self.coordinates  = self.init_coordinates(lattice_setup["train_coordinates"])
        self.YoungsModulus_E = self.graph.edata["E"][0]
        self.BeamCrossArea_A = self.graph.edata["A"][0]
        
        self.Constrainer = Constraints(self.graph.num_nodes()) # Object for adding constraints to the FE
        self.Displacer = ExternalDisplacement(self.graph.num_nodes()) # Object for enforcing node displacements (externally induced)
        
    ## START PROPERTIES ##
    @property
    def density(self):
        '''
        Calculate relative density of the lattice.
        '''
        with self.graph.local_scope():
            self.graph.apply_edges(fn.u_sub_v("coordinates", "coordinates", "delta"))
            L = (torch.norm(self.graph.edata["delta"], dim=1)).sum()

        material_height = torch.max(self.coordinates[:,1])-torch.min(self.coordinates[:,1])
        material_width = torch.max(self.coordinates[:,0])-torch.min(self.coordinates[:,0])
        material_area = material_height*material_width

        return L/material_area*np.sqrt(self.BeamCrossArea_A)
    ## END PROPERTIES ##
        
    ## START METHODS ##
    def dict_to_dgl(self,lattice_setup,option="no plot"):
        
        if option == "no plot": # graph used for actual computations
            RODS=lattice_setup["edata"]
            NODES=lattice_setup["ndata"]
            
            graph = dgl.graph(RODS["elenod"]) # generate the graph

            graph.ndata["coordinates"] = torch.stack(NODES["nodxy"]) # add information regarding nodes
            graph.ndata["constraints"] = torch.stack(NODES["nodtag"])
            graph.ndata["applied force"] = torch.stack(NODES["nodval"])
            graph.ndata["displacements"] = torch.stack(NODES["nodisp"])
            graph.edata["A"] = torch.stack(RODS["elefab"]) # add information regarding rods
            graph.edata["E"] = torch.stack(RODS["elemat"])
            graph.edata["I"] = get_BeamShapeFactor(graph.edata["A"])
            
        if option == "yes plot": # graph used for making nicer plots
            RODS=lattice_setup["edata"]
            NODES=lattice_setup["ndata"]
            SPRINGS=lattice_setup["sdata"]
            
            graph = dgl.graph(RODS["elenod"]+SPRINGS["springnod"]) # generate the graph
            graph.ndata["coordinates"] = torch.stack(NODES["nodxy"]) # add information regarding nodes
            graph.edata["edge_kind"] = torch.hstack([torch.zeros(len(RODS["elenod"])),
                                                    torch.ones(len(SPRINGS["springnod"]))])
        return graph
    
    def init_coordinates(self, train):
        '''
        Initialize the node coordinates.
        If train is True, the coordinates will be trainable using gradient descent.
        '''
        coordinates = self.graph.ndata["coordinates"]
        # if (train == True) and (coordinates.requires_grad == False):
        #     coordinates.requires_grad = True
        return coordinates  
    
    def set_constraints(self, experiment_setup):
        '''
        Set constraints of the FE simulation.
        '''
        self.Constrainer.reset_constraints()
        constraints_to_add = defaultdict(dict)
        for i in experiment_setup['forced_nodes']:
            constraints_to_add[i] = {'x': True, 'y': True}
        for i in experiment_setup['static_nodes']:
            constraints_to_add[i] = {'x': True, 'y': True}
        self.Constrainer.add_constraints_from_dict(constraints_to_add)

    def set_displacements(self, experiment_setup):
        '''
        Set external displacements for the FE simulation.
        '''
        self.Displacer.reset_displacement()
        for nodes in experiment_setup['forced_nodes']:
            self.Displacer.set_displacement(nodes, y=-experiment_setup['displacement']) 
            
    def draw_lattice(self, dr=0, ax = None, fadeout = None, path = None):
        if type(dr) == torch.Tensor:
            disp = dr.detach().numpy()
        else:
            disp = dr
        coords = self.coordinates.detach().numpy()
        ax = draw_graph(self.graph_plot.edges(),self.graph_plot.edata["edge_kind"],coords+disp,
                numbers=False, ax = ax, fadeout=fadeout)

        return ax

            
    ## END METHODS ##
