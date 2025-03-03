import numpy as np
import torch

from LattyMorph.draw import linear_fit, draw_linear_fit

import matplotlib.pyplot as plt


def get_Yprop(lattice, model, experiment_setup, draw=True):
    
    '''
    Perform a full compression experiment, with several steps where the top layer
    gets compressed further and further.

    Returns the effective elastic modulus, Poisson's ratio and the deformations for all nodes.

    Optionally, the stress-strain and strain-strain curve can be plotted.
    '''
    # collect the periferical nodes including their extremes
    all_bottom_nodes = experiment_setup["external_nodes"][0]
    all_top_nodes = experiment_setup["external_nodes"][1]
    all_left_nodes = experiment_setup["external_nodes"][2]
    all_right_nodes = experiment_setup["external_nodes"][3]
    # divide the nodes between "top and bottom", including the extremes, "left" and "right" excluding the extremes
    all_top_and_bottom_nodes = list(set(all_bottom_nodes).union(set(all_top_nodes)))
    internal_left_nodes = list(set(all_left_nodes).difference(all_top_and_bottom_nodes))
    internal_right_nodes = list(set(all_right_nodes).difference(all_top_and_bottom_nodes))
    # initialize variables per step
    total_stress_per_step = torch.zeros(experiment_setup['num_steps']+1) # Force/(Length^2)
    total_strain_per_step = torch.zeros(experiment_setup['num_steps']+1) # Length
    width_change_per_step = torch.zeros(experiment_setup['num_steps']+1) # Length
    # other variables
    draw = experiment_setup['draw_response']
    
    # compute the strain on the lattice starting from the displacement for the given number of steps
    delta = 0
    for i in range(experiment_setup['num_steps']):
        dr, strain = model(lattice, experiment_setup, delta)
        total_stress_per_step[i+1] = total_stress_per_step[i] + strain.squeeze()
        delta += dr
        width_change_per_step[i+1] = delta[internal_right_nodes,0].mean()-delta[internal_left_nodes,0].mean()
    # compute the total strain at each compression step
    total_strain_per_step = torch.arange(0, experiment_setup['num_steps']+1)*experiment_setup['displacement']

    # compute the effective Young modulus (E_s) and the poisson ratio (sigma_honey) at each compression step
    effective_modulus = total_stress_per_step.sum()/total_strain_per_step.sum() 
    poisson_ratio = width_change_per_step.sum()/total_strain_per_step.sum()

    if draw == True:
        
        fit_vars,fit_params=linear_fit(total_strain_per_step.detach().numpy(),\
                                    total_stress_per_step.detach().numpy())      
        draw_linear_fit(total_strain_per_step.detach().numpy(), 
                        total_stress_per_step.detach().numpy(),
                        effective_modulus.detach().numpy(),\
                        fit_vars, fit_params,\
                        xlabel = 'y-strain', ylabel = 'y-stress', title='Effective Young Modulus')

        fit_vars,fit_params=linear_fit(total_strain_per_step.detach().numpy(),\
                                    width_change_per_step.detach().numpy())     
        draw_linear_fit(total_strain_per_step.detach().numpy(), 
                        width_change_per_step.detach().numpy(),
                        poisson_ratio.detach().numpy(),\
                        fit_vars, fit_params,\
                        xlabel = 'y-strain', ylabel = 'x-strain', title='Poisson Ratio')
    return effective_modulus, poisson_ratio, delta