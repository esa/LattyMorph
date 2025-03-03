import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
import networkx as nx
import seaborn as sns

def draw_graph_without_coordinates(edge_list):
    '''
    Given an edge list, draw the graph without coordinates, i.e.,
    node position is random.
    '''
    # create and populate graph object
    graph = nx.Graph()
    for s,t in edge_list:
        nx.add_path(graph, [s,t])

    # draw graph
    nx.draw(graph, with_labels=True, node_color='orange')

def draw_graph(edge_list, edge_kind, coordinates, 
                numbers = True, 
                fadeout = None,
                offsetx=0.05, offsety=0.05, 
                ax = None, path = None):

    if fadeout is not None:
        node_color = fadeout#'gray'
        rod_color = fadeout#'gray'
        spring_color = fadeout#'gray'
        alpha = 1
    else:
        node_color = "skyblue"
        rod_color = "black"
        spring_color = "slategray"
        fadeout = 0.8
        alpha = 1
    node_size = 100

    if ax is None:
        fig, ax = plt.subplots()
        
    k = 0 
    for i,j in zip(edge_list[0], edge_list[1]):
        if edge_kind[k] == 0:
            ax.plot([coordinates[i][0], coordinates[j][0]],
                    [coordinates[i][1], coordinates[j][1]], 
                    color = rod_color, alpha=alpha, lw=3, zorder=1)
        if edge_kind[k] == 1:
            with matplotlib.rc_context({'path.sketch': (2.5, 10, 1)}):
                ax.plot([coordinates[i][0], coordinates[j][0]],
                        [coordinates[i][1], coordinates[j][1]], 
                        color = spring_color, alpha=alpha, zorder=1)
        k=k+1
    ax.set_xlim([-1,1.5*(coordinates)[:,0].max()])
    ax.set_ylim([-1,1.5*(coordinates)[:,1].max()])
    # plot nodes
    ax.scatter(coordinates[:, 0],coordinates[:, 1],color = node_color, alpha = alpha)
    if numbers == True:
        for i,j in zip(edge_list[0], edge_list[1]):
            # plot node labels:
            ax.text(coordinates[i][0]-offsetx, coordinates[i][1]-offsety, i)
            ax.text(coordinates[j][0]-offsetx, coordinates[j][1]-offsety, j)
    # save plot
    if path is not None:
        fig.savefig(path, bbox_inches='tight')

    return ax

def linear_fit(x,y):
    Ly = len(y)
    xfit = np.linspace(0, x[-1], Ly)
    m, b = np.polyfit(xfit, y, 1)
    return (xfit,xfit*m+b), (m,b)

def draw_linear_fit(xorig, yorig, morig, fit_vars, fit_params, xlabel='', ylabel='',title='', path = None):

    fig = plt.figure(figsize=(6,4))

    xfit,yfit=fit_vars
    m,b=fit_params
    string = "$\Delta_m$(%)=" + str(np.around((m-morig)/m,decimals=3)) + \
            "\n$\Delta_b=$"+str(np.around(b,decimals=3))

    plt.plot(xfit, yfit, color = 'k', label = 'linear fit')# linear fit
    plt.plot(xorig, yorig, color = 'darkred', linewidth = 0,\
            marker = 'o', markersize=5, label = 'fea simulation') #simulation
    plt.ylabel(ylabel, fontsize = 14)
    plt.xlabel(xlabel, fontsize = 14)
    plt.legend(fontsize=12, frameon=False)
    plt.title(title)
    plt.text(xfit[0], yfit[35], string, fontsize=12)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, bbox_inches='tight')

def draw_lattice(NODES, RODS, SPRINGS, error_handler,
        node_color="skyblue", rod_color="black", spring_color="grey", node_size=100, numbers=True, offsetx=0.05, offsety=0.05):

    if any(ele == 0 for ele in error_handler):
        print("ERROR: nodes dict or rods dict or spring dicts not compiled correctly!")
    else:
        # sort node coordinates wrt node number
        nodes=NODES["node"]
        coordinates=NODES["nodxy"]
        sorted_lists = zip(nodes, coordinates)
        sorted_lists = sorted(sorted_lists, key=lambda x: x[0])
        nodes, coordinates = zip(*sorted_lists)

        fig = plt.figure()
        # plot the rods
        for i,j in RODS["elenod"][:]:    
            plt.plot([(coordinates[i][0]).detach().numpy(),(coordinates[j][0]).detach().numpy()], 
                    [(coordinates[i][1]).detach().numpy(), (coordinates[j][1]).detach().numpy()], 
                    color = rod_color, alpha = 0.8, zorder=1, lw="3")
        # plot the springs
        for i,j in SPRINGS["springnod"][:]:    
            with matplotlib.rc_context({'path.sketch': (2.5, 10, 1)}):
                plt.plot([(coordinates[i][0]).detach().numpy(),(coordinates[j][0]).detach().numpy()], 
                        [(coordinates[i][1]).detach().numpy(), (coordinates[j][1]).detach().numpy()], 
                        color = spring_color, alpha = 0.8, zorder=1)
        # plot nodes
        plt.scatter(torch.stack(coordinates)[:,0].detach().numpy(),torch.stack(coordinates)[:,1].detach().numpy(),\
                    alpha = 1, s=node_size, marker = 'o', linewidth=0, color = node_color, zorder=2)
        # write the node number 
        if numbers == True:
            for i,j in RODS["elenod"][:]:
                plt.text(coordinates[i][0]-offsetx, coordinates[i][1]-offsety, i)
                plt.text(coordinates[j][0]-offsetx, coordinates[j][1]-offsety, j)       
        plt.xlim([-1,1.5*torch.stack(coordinates)[:,0].detach().numpy().max()])
        plt.ylim([-1,1.5*torch.stack(coordinates)[:,1].detach().numpy().max()])