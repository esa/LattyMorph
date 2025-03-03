import numpy as np 
from LattyMorph.optics.lightrays import get_all_normal_vectors, get_all_midpoints, get_all_reflected_directions, get_closest_point

def get_loss(lossf, light_direction, detector_location, model):
    '''
    Loss function used to reconfigure the Totimorphic mirror.
    '''
    # get normal vector of each unit cell in the lattice
    normals = get_all_normal_vectors(model)
    # get midpoints of all levers in the lattice
    midpoints = get_all_midpoints(model)
    # calculate light reflections given the previous information
    reflections = get_all_reflected_directions(light_direction, normals)
    
    # calculate how well the Totimorphic lattice focuses the light in a target point
    loss = 0
    for i in range(len(reflections)):
        loss += lossf(detector_location, get_closest_point(detector_location, midpoints[i], reflections[i]))
    loss /= len(reflections)
        
    return loss

def get_loss_with_damage(lossf, light_direction, detector_location, model, seed, strength=0.005):
    '''
    Loss function that includes damage to one of the mirror elements.
    '''
    normals = get_all_normal_vectors(model)
    midpoints = get_all_midpoints(model)
    reflections = get_all_reflected_directions(light_direction, normals)
    
    loss = 0
    # set seed here, so that in a single experiment, the defect stays constant
    np.random.seed(seed)
    # select which unit cell has a defect
    candidate = np.random.randint(len(reflections))
    for i in range(len(reflections)):
        add = 0
        if i == candidate:
            # add random deflection to the reflected light coming from the damaged unit cell 
            add = torch.Tensor(np.random.normal(0, 1, (3))*strength)
        refl = add+reflections[i]
        loss += lossf(detector_location, get_closest_point(detector_location, midpoints[i], refl/refl.norm()))
    loss /= len(reflections)
        
    return loss