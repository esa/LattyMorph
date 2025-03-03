import torch

class TotiBase(torch.nn.Module):
    def __init__(self):
        '''
        Base class containing common functions for all Totimorphic lattice models.
        '''
        super().__init__()

    def lattice_config(self):
        '''
        Return the configuration of the lattice, i.e., the currently set generalized coordinates / parameters.
        '''
        config = [[[] for i in range(self.shape[1])] for j in range(self.shape[0])]
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                config[i][j] = self.lattice[i][j].unitcell_config()
        return config, {'origin': self.origin.detach()}
        
    def _check_lengths(self):
        '''
        Test function. Check the length of levers and beams.
        '''
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                leverLength = (self.lattice[i][j].pointC - self.lattice[i][j].pointP).norm()
                if (leverLength - self.lattice[i][j].leverLength) > 1e-4:
                    return False
                leverLength = (self.lattice[i][j].pointC - self.lattice[i][j].pointPprime).norm()
                if (leverLength - self.lattice[i][j].leverLength) > 1e-4:
                    return False    
                beamLength = (self.lattice[i][j].pointA - self.lattice[i][j].pointB).norm()
                if (beamLength - self.lattice[i][j].beamLength) > 1e-4:
                    return False
                beamLength = (self.lattice[i][j].pointAprime - self.lattice[i][j].pointBprime).norm()
                if (beamLength - self.lattice[i][j].beamLength) > 1e-4:
                    return False          
        return True

class UnitCellBase(torch.nn.Module):
    def __init__(self):
        '''
        Base class containing common functions for all Totimorphic unitcell models.
        '''
        super().__init__()
    
    @property
    def coordinates(self):
        '''
        Return lattice coordinates.
        '''
        return [self.pointA, self.pointP, self.pointB, self.pointC, 
                self.pointPprime, self.pointAprime, self.pointBprime]
    @property
    def beamsToPlot(self):
        '''
        Return all lattice points that define the beams.
        '''
        return [[self.pointA, self.pointP, self.pointB], 
                [self.pointAprime, self.pointPprime, self.pointBprime]]

    @property
    def leverToPlot(self):
        '''
        Return all lattice points that define the levers.
        '''
        return [[self.pointP, self.pointC],
                [self.pointC, self.pointPprime]]
    
    @property
    def springsToPlot(self):    
        '''
        Return all lattice points that define the springs.
        '''
        return [[self.pointA, self.pointC],
                [self.pointB, self.pointC],
                [self.pointAprime, self.pointC],
                [self.pointBprime, self.pointC]]
