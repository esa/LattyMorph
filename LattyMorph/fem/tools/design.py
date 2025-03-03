import numpy as np

class experiment_design:
    def __init__(self,
                displacement = 0.05/50, 
                num_steps=50, 
                fit=False, 
                draw_response=True):
        '''
        Collect all options for the FEM code, i.e.,
        which nodes to use to calculate mechanical properties,
        which nodes to constrain,
        which nodes to forcefully move,
        how strongly to move them,
        how often to move them,
        etc.
        '''

        # setup experiment
        self.experiment_setup = {"forced_nodes": None, 
                                "static_nodes": None,
                                "displacement": displacement, 
                                "num_steps": num_steps,
                                "fit": fit,
                                "draw_response": draw_response,
                                "external_nodes": None} 
        # setup experiment
        self.lattice_setup =    {"type": "totimorphic",
                                "ndata": None, # nodes info
                                "edata": None, # rods info
                                "sdata": None, # spring info
                                "train_coordinates": False} # this must be false!! otherwise you cannot back-propagate
    
    ## BEGIN METHODS ##   
    
    def boundaries_nodes(self, precursor):
        """
        Returns four vectors with the labels of nodes at boundaries of the lattice.
        """
        I = precursor.verticalCellsNumber
        J = precursor.horizontalCellsNumber
        N = precursor.totalNodesNumber
        
        bottomNodesNumber = np.arange(2*J+1)
        topNodesNumber = np.arange(N-1,N-2*I-2,-1)
        leftNodesNumber = np.arange(0,(3*J+1)*I,(3*J+1))
        rightNodesNumber = np.arange(2*J,N-1,(3*J+1))
        return bottomNodesNumber, topNodesNumber, leftNodesNumber, rightNodesNumber
        
    def prepare_experiment(self, precursor, experiment_kind):
        
        if experiment_kind == "squeezing_experiment":
            
            bottom_nodes, top_nodes, left_nodes, right_nodes = self.boundaries_nodes(precursor)
            self.experiment_setup["forced_nodes"] = top_nodes
            self.experiment_setup["static_nodes"] = bottom_nodes
            self.experiment_setup["external_nodes"] = [bottom_nodes, top_nodes, left_nodes, right_nodes]
            
            self.lattice_setup["ndata"] = precursor.NODES
            self.lattice_setup["edata"] = precursor.RODS
            self.lattice_setup["sdata"] = precursor.SPRINGS
        
        return self.lattice_setup, self.experiment_setup
    
    ## END METHODS ## 