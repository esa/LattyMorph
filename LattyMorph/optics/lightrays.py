import torch
import numpy as np

def get_normal_vector(lattice):
    '''
    Get normal vector of Totimorphic unit cells (top and bottom one).
    '''
    normal_bottom = torch.cross(lattice.pointB-lattice.pointA, lattice.pointC-lattice.pointA)
    normal_top = torch.cross(lattice.pointBprime-lattice.pointC, lattice.pointPprime-lattice.pointC)
    
    return normal_bottom/normal_bottom.norm(), normal_top/normal_top.norm()

def get_all_normal_vectors(model):
    '''
    Gett normal vector of all cells in a lattice.
    '''
    normals = []
    for i in range(model.shape[0]):
        for j in range(model.shape[1]):
            new_normals = get_normal_vector(model.lattice[i][j])
            normals += new_normals
    return normals

def get_midpoint(lattice):
    '''
    Get midpoint of lever in a Totimorphic cell (again done for the top and bottom cell).
    '''
    mid_bottom, mid_top = (lattice.pointP+lattice.pointC)/2, (lattice.pointPprime+lattice.pointC)/2
    
    return mid_bottom, mid_top
    
def get_all_midpoints(model):
    '''
    Get lever midpoints of all cells in a Totimorphic lattice.
    '''
    midpoints = []
    for i in range(model.shape[0]):
        for j in range(model.shape[1]):
            new_midpoints = get_midpoint(model.lattice[i][j])
            midpoints += new_midpoints
    return midpoints

def reflected_ray_direction(light_direction, normal):
    '''
    Calculate reflection direction of a light beam with incidence 'light_direction'.
    'normal' is the normal vector of the cell that reflects the light.
    '''
    direction =  light_direction - 2 * torch.dot(light_direction, normal)*normal
    
    return direction/direction.norm()

def get_all_reflected_directions(light_direction, normals):
    '''
    Get direction of reflected light for each cell in the lattice.
    '''
    reflections = []
    for norm in normals:
        reflections.append(reflected_ray_direction(light_direction, norm))
    return reflections

def get_closest_point(target_point, origin, direction):
    '''
    Given a line (origin and direction), calculate which point on the line is closest
    to a target point.
    '''
    supp_vector = target_point - origin
    projection = torch.dot(supp_vector, direction)
    
    return origin + direction * projection

def vec(angle, radius, base):
    '''
    Returns point on a circle with radius and angle around the point 'base'.
    '''
    angle = angle/180*np.pi
    to_circle = radius*np.array([-np.cos(angle),np.sin(angle),0])
    
    return torch.Tensor(base+to_circle)