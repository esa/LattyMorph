def create_flat_2D_config(xnum, ynum):
    '''
    Returns the parameters of a flat 2D lattice.
    '''
    configuration = [[[] for i in range(ynum)] for j in range(xnum)]

    configuration[0][0] = {'phi': 0, 'phiPrime': 0, 'theta': 0.5, 'thetaPrime': 0.5}

    for i in range(xnum):
        for j in range(ynum):
            if i > 0 and j == 0:
                configuration[i][j] = {'phiPrime': 0, 'theta': 0.5, 'thetaPrime': 0.5}
            elif j > 0 and i == 0:
                configuration[i][j] = {'phi': 0, 'theta': 0.5}
            elif j > 0:
                configuration[i][j] = {'theta': 0.5}
    return configuration

def create_flat_3D_config(xnum, ynum):
    '''
    Returns the parameters of a flat 3D lattice.
    '''
    configuration = [[[] for i in range(ynum)] for j in range(xnum)]

    configuration[0][0] = {'phiz': 0, 'phiy': 0, 'thetax': 0, 'thetaxPrime': 0, 'phiyPrime': 0, 'phizPrime': 0, 'thetaz': 0.5, 'thetazPrime': 0.5}

    for i in range(xnum):
        for j in range(ynum):
            if i > 0 and j == 0:
                configuration[i][j] = {'thetax': 0, 'thetaxPrime': 0, 'phiyPrime': 0, 'phizPrime': 0, 'thetaz': 0.5, 'thetazPrime': 0.5}
            elif j > 0 and i == 0:
                configuration[i][j] = {'phiz': 0, 'phiy': 0, 'thetax': 0, 'thetaxPrime': 0, 'thetaz': 0.5, 'thetazPrime': 0.5}
            elif j > 0:
                configuration[i][j] = {'thetaz': 0.5, 'thetax': 0, 'thetazPrime': 0.5}
    return configuration
