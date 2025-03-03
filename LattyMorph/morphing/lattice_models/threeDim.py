import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
from LattyMorph.math.utils import sign, arccos_clipped
from LattyMorph.math.coordinates import polar_x, polar_y, invert_polar_x, invert_polar_y
from LattyMorph.math.rotations import rotate_x, rotate_y, rotate_z
from LattyMorph.morphing.base_models import TotiBase, UnitCellBase

EPS_3D = 1e-5

class Toti3D(TotiBase):
    def __init__(self, configuration, origin = torch.Tensor([0, 0, 0])):
        '''
        Model of Totimorphic lattice in 3 dimensions.
        '''
        super().__init__()
        self.shape = np.shape(configuration)
        
        # Create the trainable parameters
        self.config = torch.nn.ParameterList()
        for i in range(self.shape[0]):
            columns = torch.nn.ParameterList()
            for j in range(self.shape[1]):
                ucell_dict = {}
                for keys in configuration[i][j].keys():
                    ucell_dict[keys] = torch.Tensor([configuration[i][j][keys]*np.pi])
                columns.append(torch.nn.ParameterDict(ucell_dict))
            self.config.append(columns)
        self.origin= torch.nn.Parameter(origin)
            
        # Construct the lattice from UnitCell elements (unitcell by unitcell)
        self.lattice = [[[] for i in range(self.shape[1])] for j in range(self.shape[0])]
        self.lattice[0][0] = UnitCell(phiz = self.config[0][0]['phiz'][0], 
                                      phiy = self.config[0][0]['phiy'][0],
                                      thetaz = self.config[0][0]['thetaz'][0],
                                      thetax = self.config[0][0]['thetax'][0],
                                      phizPrime = self.config[0][0]['phizPrime'][0], 
                                      phiyPrime = self.config[0][0]['phiyPrime'][0],
                                      thetazPrime = self.config[0][0]['thetazPrime'][0],
                                      thetaxPrime = self.config[0][0]['thetaxPrime'][0],
                                      pointA = self.origin)
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if i > 0 and j == 0:
                    self.lattice[i][j] = UnitCell(phiz = None, phiy = None,
                                                  thetaz = self.config[i][j]['thetaz'][0], 
                                                  thetax = self.config[i][j]['thetax'][0], 
                                                  phizPrime = self.config[i][j]['phizPrime'][0], 
                                                  phiyPrime = self.config[i][j]['phiyPrime'][0],
                                                  thetazPrime = self.config[i][j]['thetazPrime'][0],
                                                  thetaxPrime = self.config[i][j]['thetaxPrime'][0])
                elif j > 0 and i == 0:
                    self.lattice[i][j] = UnitCell(phiz = self.config[i][j]['phiz'][0], 
                                                  phiy = self.config[i][j]['phiy'][0],
                                                  thetaz = self.config[i][j]['thetaz'][0],
                                                  thetax = self.config[i][j]['thetax'][0],
                                                  thetazPrime = self.config[i][j]['thetazPrime'][0])
                elif j > 0:
                    self.lattice[i][j] = UnitCell(phiz = None, phiy = None,
                                                  thetaz = self.config[i][j]['thetaz'][0],
                                                  thetax = self.config[i][j]['thetax'][0],
                                                  thetazPrime = self.config[i][j]['thetazPrime'][0])
                    
    def forward(self):
        '''
        Forward is recalculating the points of each unit cell.
        Follows the same order used when initially constructing the lattice.
        '''
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if i == 0 and j == 0:
                    integrity = self.lattice[i][j].forward(pointA=self.origin)
                elif i > 0 and j == 0:
                    integrity = self.lattice[i][j].forward(phiz = self.lattice[i-1][j].phizPrime, 
                                                           phiy = self.lattice[i-1][j].phiyPrime, 
                                               pointA = self.lattice[i-1][j].pointAprime,
                                               pointB = self.lattice[i-1][j].pointBprime) 
                elif j > 0 and i == 0:
                    integrity = self.lattice[i][j].forward(pointA = self.lattice[i][j-1].pointB, 
                                               pointAprime = self.lattice[i][j-1].pointBprime)
                elif j > 0:
                    integrity = self.lattice[i][j].forward(phiz = self.lattice[i-1][j].phizPrime, 
                                               phiy = self.lattice[i-1][j].phiyPrime, 
                                               pointA = self.lattice[i][j-1].pointB, 
                                               pointAprime = self.lattice[i][j-1].pointBprime,
                                               pointB = self.lattice[i-1][j].pointBprime) 
            if integrity == False:
                return False
        
        return True
        
    def plot(self, xlim, ylim, zlim, ax = None, alpha = None, msize=5, sketch = (2.5, 10, 1)):
        '''
        Plot the lattice. This is done by calling the plot function of each individual cell.
        '''
        if ax is None:
            ax = self.lattice[0][0].plot(xlim=xlim, ylim=ylim, zlim=zlim, alpha = alpha, msize=msize, sketch=sketch)
        else:
            self.lattice[0][0].plot(xlim=xlim, ylim=ylim, zlim=zlim, ax=ax, alpha = alpha, msize=msize, sketch=sketch)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if i != 0 or j != 0:
                    self.lattice[i][j].plot(ax=ax, alpha = alpha, msize=msize, sketch=sketch)
        return ax


class UnitCell(UnitCellBase):
    def __init__(self, phiz, phiy, thetaz, thetax, phizPrime = None, phiyPrime = None, thetazPrime = None, thetaxPrime = None, pointA = None):
        '''
        A single Totimorphic elementary cell (named unit cell here).

        Contains all methods to calculate, from constraints and parameters, the physical lattice coordinates.
        '''
        super().__init__()
        self.beamLength = 1.
        self.leverLength = self.beamLength/2.
        self.pointA = pointA
        
        self.phiy = phiy
        self.phiz = phiz
        self.thetax = thetax
        self.thetaz = thetaz
        
        self.phiyPrime = phiyPrime
        self.phizPrime = phizPrime
        self.thetaxPrime = thetaxPrime
        self.thetazPrime = thetazPrime
        
    def forward(self, pointA = None, phiz = None, phiy = None, pointAprime = None, pointB = None):
        '''
        Calculate physical coordinates of the cell.

        Constraints (i.e., fixing point positions or beam angles) can be set.
        '''
        if pointA is not None:
            self.pointA = pointA
        if phiz is not None:
            self.phiz = phiz
        if phiy is not None:
            self.phiy = phiy
        
        if pointB is None:
            linePB = self.calc_linePB()
        else:
            linePB = (pointB-self.pointA)/self.beamLength  # NEW
        
        self.pointB = self.calc_pointB(linePB)
        self.pointP = self.calc_pointP(linePB)

        if pointAprime is None:
            linePC = self.calc_linePC(self.thetaz)
        
            self.pointC = self.calc_pointC(linePC)
            
            lineCPprime = self.calc_lineCPprime()

            self.pointPprime = self.calc_pointPprime(lineCPprime)

            linePprimeBprime = self.calc_linePprimeBprime()

            self.pointAprime = self.calc_pointAprime(linePprimeBprime)
            self.pointBprime = self.calc_pointBprime(linePprimeBprime)
        else:
            self.pointAprime = pointAprime
            
            rtilde = self.pointAprime - self.pointP
            rtilde = rotate_y(-self.phiy).matmul(rtilde)
            rtilde = rotate_z(-self.phiz).matmul(rtilde)  
            rtilde = rotate_x(-self.thetax).matmul(rtilde)
            
            full_norm, theta_r, phi_r = invert_polar_x(rtilde)
            C = (full_norm**2-3/4 * self.beamLength**2)/(full_norm*self.beamLength*torch.sin(theta_r))
            if C >= 1:
                return False
            Cangle = arccos_clipped(C)
            lower_min = phi_r - Cangle
            upper_min = phi_r + Cangle
            
            if phi_r < 0:
                return False
            actual_thetaz = torch.clamp(self.thetaz, lower_min+EPS_3D, upper_min-EPS_3D)
            linePC = self.calc_linePC(actual_thetaz)
            self.pointC = self.calc_pointC(linePC)
            
            rnorm, alpha, beta = self.calc_springVec()
            if torch.sin(beta).abs() <= EPS_3D:
                return False
            else:
                range_param = arccos_clipped(-rnorm/self.beamLength)
                lower_range = -beta+range_param
                upper_range = 2*np.pi - beta - range_param 
                theta_actual = torch.clamp(self.thetazPrime, min=lower_range+EPS_3D, max=upper_range-EPS_3D)
                
                Cfac = -(rnorm/self.beamLength + torch.cos(beta)*torch.cos(theta_actual))/(torch.sin(beta)*torch.sin(theta_actual))
                if Cfac > 0:
                    lower_range = beta-range_param
                    upper_range = beta + range_param 
                    theta_actual = torch.clamp(self.thetazPrime, min=lower_range+EPS_3D, max=upper_range-EPS_3D)

                    Cfac = -(rnorm/self.beamLength + torch.cos(beta)*torch.cos(theta_actual))/(torch.sin(beta)*torch.sin(theta_actual))
                    print(Cfac)
                if Cfac.abs() >= 1:
                    return False
                
                phi_actual = alpha - arccos_clipped(Cfac)

            unitVec = polar_x(theta_actual, phi_actual)
            _, self.phizPrime, self.phiyPrime = invert_polar_y(unitVec)
                        
            self.pointPprime = self.calc_pointPprime(None, unitVec)
            self.pointBprime = self.calc_pointBprime(None, unitVec)
        return True
            
    # The following are helper functions for calculating the points.
    def calc_linePB(self):
        return polar_y(self.phiz, self.phiy)
    
    def calc_linePC(self, thetaz):
        res = torch.Tensor([1,0,0]) 
        res = rotate_z(thetaz).matmul(res) 
        res = rotate_x(self.thetax).matmul(res)
        res = rotate_z(self.phiz).matmul(res)
        res = rotate_y(self.phiy).matmul(res)
        
        return res
    
    def calc_pointB(self, linePB):
        return self.pointA + self.beamLength*linePB
    
    def calc_pointP(self, linePB):
        return self.pointA + self.leverLength*linePB
    
    def calc_pointC(self, linePC):
        return self.pointP + self.leverLength*linePC
    
    def calc_lineCPprime(self):
        res = torch.Tensor([1,0,0])
        res = rotate_z(-self.thetazPrime).matmul(res) 
        res = rotate_x(-self.thetaxPrime).matmul(res) 
        res = rotate_z(self.phizPrime).matmul(res) 
        res = rotate_y(self.phiyPrime).matmul(res) 
        
        return res
    
    def calc_pointPprime(self, lineCPprime, lineAprimePprime = None):
        if lineAprimePprime is None:
            return self.pointC - self.leverLength*lineCPprime
        else:
            return self.pointAprime + self.leverLength*lineAprimePprime
    
    def calc_linePprimeBprime(self):
        return polar_y(self.phizPrime, self.phiyPrime)
    
    def calc_pointAprime(self, linePprimeBprime):
        return self.pointPprime - self.leverLength*linePprimeBprime
    
    def calc_pointBprime(self, linePprimeBprime, lineAprimePprime = None):
        if lineAprimePprime is None:
            return self.pointPprime + self.leverLength*linePprimeBprime
        else:
            return self.pointAprime + self.beamLength*lineAprimePprime
    
    def calc_springVec(self):
        springVec = self.pointAprime - self.pointC
        springVec_norm, theta, phi = invert_polar_x(springVec)
    
        return springVec_norm, phi, theta
    
    def unitcell_config(self):
        config = {'phiz': float(self.phiz.detach().numpy())/np.pi, 
                'phizPrime': float(self.phizPrime.detach().numpy())/np.pi, 
                'phiy': float(self.phiy.detach().numpy())/np.pi, 
                'phiyPrime': float(self.phiyPrime.detach().numpy())/np.pi, 
                'thetaz': float(self.thetaz.detach().numpy())/np.pi, 
                'thetazPrime': float(self.thetazPrime.detach().numpy())/np.pi,
                'thetax': float(self.thetax.detach().numpy())/np.pi}
        if self.thetaxPrime is not None:
            config['thetaxPrime'] = float(self.thetaxPrime.detach().numpy())/np.pi
        return config
    
    def plot(self, xlim = None, ylim = None, zlim = None, ax = None, alpha = None, msize=5, sketch = None):
        '''
        Plot a single cell.
        '''
        if ax is None:
            ax = plt.axes(projection='3d') #= #plt.subplots()
            
        for points in self.leverToPlot:
            xcoords = []
            ycoords = []
            zcoords = []
            for pt in points:
                xcoords.append(pt[0].detach())
                ycoords.append(pt[1].detach())
                zcoords.append(pt[2].detach())
            ax.plot3D(xcoords, ycoords, zcoords, markersize=msize, marker='o', color='steelblue')
        
        for points in self.springsToPlot:
            xcoords = []
            ycoords = []
            zcoords = []
            for pt in points:
                xcoords.append(pt[0].detach())
                ycoords.append(pt[1].detach())
                zcoords.append(pt[2].detach())
            with matplotlib.rc_context({'path.sketch': sketch}):
                ax.plot3D(xcoords, ycoords, zcoords, markersize=msize, marker='o', color='slategray')
            
        for points in self.beamsToPlot:
            xcoords = []
            ycoords = []
            zcoords = []
            for pt in points:
                xcoords.append(pt[0].detach())
                ycoords.append(pt[1].detach())
                zcoords.append(pt[2].detach())
            ax.plot3D(xcoords, ycoords, zcoords, markersize=msize, marker='o', color='k')
                
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if zlim is not None:
            ax.set_zlim(zlim)
            
        return ax