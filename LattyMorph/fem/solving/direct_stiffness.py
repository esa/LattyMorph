import numpy as np
import torch
import torch.nn as nn

import dgl
import dgl.function as fn

from collections import defaultdict

from LattyMorph.fem.mapping import ExternalForces

class StiffnessMatrix(nn.Module):
    def __init__(self):
        '''
        Construct the stiffness matrix of a given lattice.
        '''
        super().__init__()

    def forward(self, graph, delta):
        '''
        Returns the stiffness matrix given a DGL graph representing the lattice.
        '''
        with graph.local_scope():
            
            # squeeze vectors. 
            E=graph.edata["E"].squeeze()
            A=graph.edata["A"].squeeze()
            I=graph.edata["I"].squeeze()
            
            # update coordinates
            graph.ndata['coordinates_upd'] = graph.ndata['coordinates'] + delta
            
            # get node distances
            graph.apply_edges(fn.u_sub_v("coordinates_upd", "coordinates_upd", "delta"))
            
            # calculate rod lengths and stiffness constants
            L = torch.norm(graph.edata["delta"], dim=1)
            graph.edata["krot"] = ((E*I)/(L**3))
            graph.edata["klin"] = ((E*A)/L)

            # calculate angles
            cos = (graph.edata["delta"][:,0]/L).view(-1,1)
            sin = -(graph.edata["delta"][:,1]/L).view(-1,1)
            L = L.view(-1,1)

            # factors that appear many times in the stiffness matrix
            sinsq = sin**2
            cossq = cos**2
            sincos = sin*cos
            Lsin = 6*L*sin
            Lcos = 6*L*cos
            L2 = 2*L**2
            L4 = 4*L**2

            Kmat_rot = torch.cat([12*sinsq,  12*sincos, -Lsin, -12*sinsq,   -12*sincos, -Lsin,\
                                 12*sincos,  12*cossq,   -Lcos,  -12*sincos, -12*cossq,  -Lcos,\
                                 -Lsin,       -Lcos,       L4,    Lsin,       Lcos,      L2,\
                                 -12*sinsq,   -12*sincos,  Lsin,  12*sinsq,  12*sincos,  Lsin,\
                                  -12*sincos, -12*cossq,  Lcos, 12*sincos,  12*cossq,  Lcos,\
                                 -Lsin,       -Lcos,      L2,     Lsin,       Lcos,      L4],
                                 dim=1)

            # apply prefactor
            Kmat_rot = Kmat_rot*graph.edata["krot"].view(-1,1)

            # Truss rod element
            zeros = torch.zeros((len(graph.edges()[0]),1))
            Kmat_lin = torch.cat([cossq,  -sincos, zeros, -cossq,  sincos, zeros,\
                                  -sincos, sinsq,  zeros, sincos, -sinsq,  zeros,\
                                  zeros,  zeros,  zeros,  zeros,   zeros,  zeros,\
                                 -cossq, sincos, zeros,  cossq,   -sincos, zeros,\
                                 sincos,-sinsq,  zeros,  -sincos,  sinsq,  zeros,\
                                  zeros,  zeros,  zeros,  zeros,   zeros,  zeros],
                                 dim=1)
            # apply prefactor
            Kmat_lin = Kmat_lin*graph.edata['klin'].view(-1,1)

            # add both to get generalised beam element
            Kmat = Kmat_rot+Kmat_lin
            # reshape to get 6x6 matrix for each beam element
            Kmat = Kmat.view(-1,6,6)

        # combine stiffness matrices of individual beams
        # to get the full stiffness matrix
        fullmat = torch.zeros((graph.num_nodes()*3, graph.num_nodes()*3))
        for i in range(len(graph.edges()[0])):
            n0, n1 = graph.edges()[0][i], graph.edges()[1][i]
            edgeK = Kmat[i]
            for j in range(3):
                fullmat[n0*3+j, n0*3:(n0+1)*3] += edgeK[j][:3]
                fullmat[n0*3+j, n1*3:(n1+1)*3] += edgeK[j][3:]
            for j in range(3):
                fullmat[n1*3+j, n0*3:(n0+1)*3] += edgeK[j+3][:3]
                fullmat[n1*3+j, n1*3:(n1+1)*3] += edgeK[j+3][3:]

        return fullmat


class DirectStiffnessSolver(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, latticeGdl, stiffness, experiment_setup):
        '''
        Given a lattice and its stiffness matrix, perform a single compression step.

        Returns the deformations (dr) and stress.
        '''
        latticeGdl.set_constraints(experiment_setup)
        latticeGdl.set_displacements(experiment_setup)
        internal_forces = latticeGdl.Displacer.get_internal_forces(stiffness)

        forces_to_set = defaultdict(dict)
        for i in range(latticeGdl.graph.num_nodes()):
            forces_to_set[i] = {'Fx': internal_forces[i*3], 'Fy': internal_forces[i*3+1], 'Mphi': internal_forces[i*3+2]}
        forces = ExternalForces(latticeGdl.graph.num_nodes(), forces_to_set)

        reduced_stiffness = latticeGdl.Constrainer.transform_stiffness_matrix(stiffness)
        reduced_force = latticeGdl.Constrainer.transform_force_vector(forces.force_vector)

        disp = torch.linalg.solve(reduced_stiffness, reduced_force)
        assert(np.isnan(disp.detach().numpy()).any()==False)

        disp = latticeGdl.Constrainer.reverse_transform_displacements(disp)+latticeGdl.Displacer.displacement
        dr = torch.reshape(disp, (latticeGdl.graph.num_nodes(), 3))[:,:2]

        top_force = torch.matmul(stiffness, disp)
        top_force = torch.reshape(top_force, (latticeGdl.graph.num_nodes(), 3))[experiment_setup['forced_nodes']][:,1]
        stress = -torch.sum(top_force)/np.sqrt(latticeGdl.BeamCrossArea_A[0])
        return dr, stress
