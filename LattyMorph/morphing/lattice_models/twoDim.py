import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
from LattyMorph.math.utils import sign, arccos_clipped
from LattyMorph.math.coordinates import unitVector
from LattyMorph.morphing.base_models import TotiBase, UnitCellBase

EPS_2D = 1e-7

class Toti2D(TotiBase):
    def __init__(self, configuration, origin = torch.Tensor([0, 0])):
        '''
        Model of Totimorphic lattice in 2 dimensions.
        '''
        super().__init__()
        self.shape = np.shape(configuration)
        # print(configuration)
        
        self.config = torch.nn.ParameterList()
        # print(self.config)
        for i in range(self.shape[0]):
            columns = torch.nn.ParameterList()
            for j in range(self.shape[1]):
                ucell_dict = {}
                for keys in configuration[i][j].keys():
                    ucell_dict[keys] = torch.Tensor([configuration[i][j][keys]*np.pi])
                columns.append(torch.nn.ParameterDict(ucell_dict))
            self.config.append(columns)  
        self.origin= torch.nn.Parameter(origin)
        # print(self.config[0][0]['phi'][0])
            
        self.lattice = [[[] for i in range(self.shape[1])] for j in range(self.shape[0])]
        self.lattice[0][0] = UnitCell(self.config[0][0]['phi'][0], 
                                      self.config[0][0]['theta'][0], 
                                      pointA = self.origin,
                                      phiPrime=self.config[0][0]['phiPrime'][0], 
                                      thetaPrime=self.config[0][0]['thetaPrime'][0])  
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if i > 0 and j == 0:
                    self.lattice[i][j] = UnitCell(None,   
                                                  self.config[i][j]['theta'][0], 
                                                  phiPrime = self.config[i][j]['phiPrime'][0], 
                                                  thetaPrime=self.config[i][j]['thetaPrime'][0])
                elif j > 0 and i == 0:
                    self.lattice[i][j] = UnitCell(self.config[i][j]['phi'][0], 
                                                  self.config[i][j]['theta'][0])
                elif j > 0:
                    self.lattice[i][j] = UnitCell(None,   
                                                  self.config[i][j]['theta'][0])
                    
    def forward(self):
        '''
        Forward is recalculating the points of each unit cell.
        Follows the same order used when initially constructing the lattice.
        '''
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if i == 0 and j == 0:
                    integrity = self.lattice[i][j].forward(pointA = self.origin)
                elif i > 0 and j == 0:
                    integrity = self.lattice[i][j].forward(phi = self.lattice[i-1][j].phiPrime,   
                                               pointA = self.lattice[i-1][j].pointAprime)
                elif j > 0 and i == 0:
                    integrity = self.lattice[i][j].forward(pointA = self.lattice[i][j-1].pointB, 
                                               pointAprime = self.lattice[i][j-1].pointBprime)
                elif j > 0:
                    integrity = self.lattice[i][j].forward(phi = self.lattice[i-1][j].phiPrime,   
                                               pointA = self.lattice[i][j-1].pointB, 
                                               pointAprime = self.lattice[i][j-1].pointBprime)
            if integrity == False:
                return False
        
        return True
        
    def plot(self, xlim, ylim, ax = None, alpha = None):
        '''
        Plot the lattice. This is done by calling the plot function of each individual cell.
        '''
        if ax is None:
            ax = self.lattice[0][0].plot(xlim=xlim, ylim=ylim, alpha = alpha)
        else:
            self.lattice[0][0].plot(xlim=xlim, ylim=ylim, ax=ax, alpha = alpha)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if i != 0 or j != 0:
                    self.lattice[i][j].plot(ax=ax, alpha = alpha)
        return ax

                
class UnitCell(UnitCellBase):
    
    def __init__(self, phi = None, theta = None, phiPrime = None, thetaPrime = None, pointA = None):
        '''
        A single Totimorphic elementary cell (named unit cell here).

        Contains all methods to calculate, from constraints and parameters, the physical lattice coordinates.
        '''
        super().__init__()
        self.beamLength = 1.
        self.leverLength = self.beamLength/2.
        self.phi = phi
        self.phiPrime = phiPrime
        self.theta = theta
        self.thetaPrime = thetaPrime

        self.pointA = pointA
        self.pointP = None
        self.pointB = None
        self.pointC = None
        self.pointPprime = None
        self.pointAprime = None
        self.pointBprime = None
        
    def forward(self, phi = None, pointA = None, pointAprime = None):
        '''
        Calculate physical coordinates of the cell.

        Constraints (i.e., fixing point positions or beam angles) can be set.
        '''
        if pointA is not None:
            self.pointA = pointA
        if phi is not None:
            self.phi = phi
        self.pointP = self.calc_pointP()
        self.pointB = self.calc_pointB()
        
        if pointAprime is not None:
            self.pointAprime = pointAprime
            theta, theta_criteria = self.clamp_theta_to_allowed_range()
            if theta_criteria == False:
                return False # unit cell invalid
        else:
            theta = self.theta
        self.theta_limited = theta
        
        self.pointC = self.calc_pointC(theta)
        self.pointPprime, springLengthCriteria = self.calc_pointPprime(pointAprime)
        if springLengthCriteria == False:
            return False # unit cell invalid
        
        if pointAprime is None:
            self.pointAprime = self.calc_pointAprime()
            
        self.pointBprime = self.calc_pointBprime()
        
        if pointAprime is not None:
            self.phiPrime = self.calc_phiPrime()
            self.thetaPrime = self.calc_thetaPrime(self.phiPrime)
            
        overlapCriteria = self.springs_not_crossing(theta)
        if overlapCriteria == False:
            return False # unit cell invalid
        
        return True # unit cell valid and constructed, yay!
    
    def unitcell_config(self):
        return {'phi': float(self.phi.detach().numpy())/np.pi, 
                'phiPrime': float(self.phiPrime.detach().numpy())/np.pi, 
                'theta': float(self.theta_limited.detach().numpy())/np.pi, 
                'thetaPrime': float(self.thetaPrime.detach().numpy())/np.pi}
    
    # The following are helper functions for calculating the points.
    def calc_pointP(self):
        return self.pointA + self.leverLength*unitVector(self.phi)
    
    def calc_pointB(self):
        return self.pointA + self.beamLength*unitVector(self.phi)
    
    def calc_pointC(self, theta):
        return self.pointP + self.leverLength*unitVector(self.phi+theta)
    
    def calc_pointPprime(self, pointAprime):
        if pointAprime is None:
            return self.pointC + self.leverLength*unitVector(np.pi-self.thetaPrime+self.phiPrime), True
        else:
            delta = 0.5*(self.pointAprime.norm()**2 - self.pointC.norm()**2)
            Aplus = self.pointAprime[1]+self.pointC[1]
            A0 = self.pointAprime[0]-self.pointC[0]
            A1 = self.pointAprime[1]-self.pointC[1]
            Anorm = (self.pointAprime-self.pointC).norm()
            sqrterm = A0*torch.sqrt((self.beamLength/Anorm)**2 - 1)
            G1 = (Aplus - sqrterm)*0.5
            G0 = (delta-G1*A1)/A0
            res = torch.Tensor([0,0])
            res[0] = G0
            res[1] = G1
            return res, self.beamLength/Anorm-1 >= 0
        
    def calc_pointAprime(self):
        return self.pointPprime - self.leverLength*unitVector(self.phiPrime)
        
    def calc_pointBprime(self):
        return 2*self.pointPprime - self.pointAprime
    
    def calc_phiPrime(self):
        vector = (self.pointBprime-self.pointAprime)/self.beamLength
        phiPrime = sign(vector[1])*arccos_clipped(vector[0])
        return phiPrime
        
    def calc_thetaPrime(self, phiPrime):
        vector = (self.pointPprime - self.pointC)/self.leverLength
        gamma = arccos_clipped(vector[0])
        if vector[1] < 0:
            gamma = -gamma
        return np.pi - gamma + phiPrime
    
    def calc_thetaRange(self):
        rvec = self.pointAprime-self.pointA-self.leverLength*unitVector(self.phi)
        rnorm = rvec.norm()
        rangle = sign(rvec[1])*arccos_clipped(torch.dot(rvec, unitVector(self.phi*0.))/rnorm)
                 
        delta = rangle - self.phi
        lower = rnorm/self.beamLength - 3/4. * self.beamLength/rnorm
        return (delta-arccos_clipped(lower))/np.pi, (delta+arccos_clipped(lower))/np.pi
    
    def clamp_theta_to_allowed_range(self):
        theta_min, theta_max = self.calc_thetaRange()
        theta = torch.clamp(self.theta, theta_min*np.pi+EPS_2D, theta_max*np.pi-EPS_2D)
        theta_criteria = (torch.isnan(theta_min) == False and torch.isnan(theta_max) == False)
        
        return theta, theta_criteria
    
    def springs_not_crossing(self, theta):
        '''
        Check for collisions between springs.
        '''
        alpha = self.thetaPrime/2-self.phiPrime
        alphaP = np.pi/2 - alpha
        beta = self.phi+1/2*theta
        betaP = np.pi/2 - beta
        
        totalAngle0 = alpha+beta
        totalAngle1 = alphaP+betaP
    
        if totalAngle0 < 0:
            return False
        if totalAngle1 < 0:
            return False
        
        return True
    
    def add_node_labels(self, zero_counter = 0, pointA = None, pointP = None, pointB = None, pointAprime = None):
        pass 
    
    def plot(self, xlim = None, ylim = None, ax = None, alpha = None):
        '''
        Plot a single cell.
        '''
        if ax is None:
            fig, ax = plt.subplots()
            
        if alpha is None:
            leverColor = 'steelblue'
            springColor = 'slategray'
            beamColor = 'k'
            alpha = 1
        else:
            leverColor = 'k'
            springColor = 'k'
            beamColor = 'k'
            
        for points in self.leverToPlot:
            xcoords = []
            ycoords = []
            for pt in points:
                xcoords.append(pt[0].detach())
                ycoords.append(pt[1].detach())
            ax.plot(xcoords, ycoords, markersize=5, marker='o', color=leverColor, alpha = alpha)
        
        for points in self.springsToPlot:
            xcoords = []
            ycoords = []
            for pt in points:
                xcoords.append(pt[0].detach())
                ycoords.append(pt[1].detach())
            with matplotlib.rc_context({'path.sketch': (2.5, 10, 1)}):
                ax.plot(xcoords, ycoords, markersize=5, marker='o', color=springColor, alpha = alpha)
            
        for points in self.beamsToPlot:
            xcoords = []
            ycoords = []
            for pt in points:
                xcoords.append(pt[0].detach())
                ycoords.append(pt[1].detach())
            ax.plot(xcoords, ycoords, markersize=5, marker='o', color=beamColor, alpha = alpha)
                
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
            
        return ax
