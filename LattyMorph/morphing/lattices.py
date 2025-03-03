from LattyMorph.morphing.configs import create_flat_2D_config, create_flat_3D_config
from LattyMorph.morphing.lattice_models import Toti2D
from LattyMorph.morphing.lattice_models import Toti3D
import torch

def create_flat_2D_sheet(num_rows, num_columns):
    '''
    Helper function to initialize a 2D Totimorphic model in its flat configuration.
    '''
    config = create_flat_2D_config(num_rows, num_columns)
    model = Toti2D(config)
    model.forward()
    return model

def create_flat_3D_sheet(num_rows, num_columns):
    '''
    Helper function to initialize a 3D Totimorphic model in its flat configuration.
    '''
    config = create_flat_3D_config(num_rows, num_columns)
    origin = torch.Tensor([0,0,0])
    model = Toti3D(config, origin=origin)
    model.forward()
    return model