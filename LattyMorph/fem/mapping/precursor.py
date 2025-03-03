import torch
from LattyMorph.draw import draw_lattice

class precursor():
    def __init__(self,model2D,properties):
        '''
        Given a Totimorphic lattice object (represented by lattice parameters and lattice physical coordinates),
        collect all information required to perform a finite element experiment.

        In principle, this is the interface between the Totimorphic model and the FEM code (using DGL graphs).
        '''
        
        # trainable pytorch model
        self.model2D = model2D 
        
        # rods properties
        self.A = properties["beam_cross_section"] 
        self.E = properties["beam_young_module"]

        # lattice characterizing dimensions
        self.I = self.model2D.shape[0] # number of rows
        self.J = self.model2D.shape[1] # number of columns
        self.nodesNumber = (2*self.J+1)*(self.I+1)+self.I*self.J #total number of nodes
        self.rodsNumber = (2*self.J)*(self.I+1)+(2*self.J)*self.I #total number of rods
        self.springNumber = 4*self.I*self.J
        
        # error handler variables
        self.NODES_COMPILED = 0
        self.RODS_COMPILED = 0
        self.SPRINGS_COMPILED = 0
        
    ## START PROPERTIES ##
    
    @property
    def verticalCellsNumber(self):
        return self.I
    @property
    def horizontalCellsNumber(self):
        return self.J
        
    @property
    def totalNodesNumber(self):
        return self.nodesNumber
        
    @property
    def totalRodsNumber(self):
        return self.rodsNumber 
    
    @property
    def totalSpringNumber(self):
        return self.springNumber
    
    ## END PROPERTIES ##

    ## START METHOD DEFINITIONS ##

    def forward(self):
        """
        Initialize the pytorch model Toti2D
        """
        return self.model2D.forward()
    
    def initialize(self):
        """
        Initialize and compile the dictionaries containing the nodes, rods, springs mapping and properties
        """
        self.NODES = { "node": [], "nodxy": [], "nodtag": [], "nodval": [], "nodisp": [] }
        self.RODS = { "element": [], "elenod": [], "elefab": [], "elemat": [] }
        self.SPRINGS = { "spring": [], "springnod": [] }
        self.compile_nodes_dict()
        self.compile_rods_dict()
        self.compile_springs_dict()
        if any(ele == 0 for ele in [self.NODES_COMPILED,self.RODS_COMPILED,self.SPRINGS_COMPILED]):
            print("compiling problem, good luck!")
        
    def M00(self,i,j):
        """
        Assign the nodes coordinates and numbering in cell i=0, j=0
        """
        # lower row - nodes.
        self.NODES["node"].append(0)
        self.NODES["nodxy"].append(self.model2D.lattice[i][j].pointA)
        self.NODES["node"].append(1)
        self.NODES["nodxy"].append(self.model2D.lattice[i][j].pointP)
        self.NODES["node"].append(2)
        self.NODES["nodxy"].append(self.model2D.lattice[i][j].pointB)
        
        # middle row.
        self.NODES["node"].append(2*self.J+1)
        self.NODES["nodxy"].append(self.model2D.lattice[i][j].pointC)

        # upper row
        self.NODES["node"].append(3*self.J+1)
        self.NODES["nodxy"].append(self.model2D.lattice[i][j].pointAprime)
        self.NODES["node"].append(3*self.J+2)
        self.NODES["nodxy"].append(self.model2D.lattice[i][j].pointPprime)
        self.NODES["node"].append(3*self.J+3)
        self.NODES["nodxy"].append(self.model2D.lattice[i][j].pointBprime)

    def M0X(self,i,j):
        """
        Assign the nodes coordinates and numbering in cell i=0, j=1,..,M
        """
        # lower row.
        self.NODES["node"].append(2*j+1)
        self.NODES["nodxy"].append(self.model2D.lattice[i][j].pointP)
        self.NODES["node"].append(2*j+2)
        self.NODES["nodxy"].append(self.model2D.lattice[i][j].pointB)

        # middle row.
        self.NODES["node"].append(2*self.J+j+1)
        self.NODES["nodxy"].append(self.model2D.lattice[i][j].pointC)    

        # upper row
        self.NODES["node"].append(3*self.J+2*j+2)
        self.NODES["nodxy"].append(self.model2D.lattice[i][j].pointPprime)
        self.NODES["node"].append(3*self.J+2*j+3)
        self.NODES["nodxy"].append(self.model2D.lattice[i][j].pointBprime)

    def MX0(self,i,j):
        """
        Assign the nodes coordinates and numbering in cell i=1,..,N, j=0
        """
        # middle row.
        self.NODES["node"].append((i+1)*(2*self.J+1)+(i*self.J))
        self.NODES["nodxy"].append(self.model2D.lattice[i][j].pointC)

        # upper row
        self.NODES["node"].append((i+1)*(2*self.J+1)+(i+1)*self.J)
        self.NODES["nodxy"].append(self.model2D.lattice[i][j].pointAprime)
        self.NODES["node"].append((i+1)*(2*self.J+1)+(i+1)*self.J+1)
        self.NODES["nodxy"].append(self.model2D.lattice[i][j].pointPprime)
        self.NODES["node"].append((i+1)*(2*self.J+1)+(i+1)*self.J+2)
        self.NODES["nodxy"].append(self.model2D.lattice[i][j].pointBprime)

    def MXX(self,i,j):
        """
        Assign the nodes coordinates and numbering in cell i=1,..,N, j=1,..,M
        """        
        # middle row
        self.NODES["node"].append((i+1)*(2*self.J+1)+(i*self.J)+j)
        self.NODES["nodxy"].append(self.model2D.lattice[i][j].pointC)

        # upper row.
        self.NODES["node"].append((i+1)*(2*self.J+1)+(i+1)*self.J+2*j+1)
        self.NODES["nodxy"].append(self.model2D.lattice[i][j].pointPprime)
        self.NODES["node"].append((i+1)*(2*self.J+1)+(i+1)*self.J+2*j+2)
        self.NODES["nodxy"].append(self.model2D.lattice[i][j].pointBprime)
        
    def fields_compiler(self, field, nodeAttribute):
        """
        Assign potential forces, displacements and constraints to the nodes
        """
        if field == None:
            for _ in self.NODES["node"]:
                self.NODES[nodeAttribute].append(torch.tensor([0,0], dtype=torch.float32))
        else:
            self.NODES[nodeAttribute] = field

    def compile_nodes_dict(self, constraintField=None, forceField=None, dispField=None):
        """
        Compile the dictionary relative to nodes
        """
        for i in range(self.I):
            for j in range(self.J): 
                if i==0 and j==0:
                    self.M00(i,j) #explore the first cell.
                if i==0 and j>0:
                    self.M0X(i,j) #explore the first row.
                if  i>0 and j==0:
                    self.MX0(i,j) #explore the first column.
                if i>0 and j>0:
                    self.MXX(i,j) #explore the central part of the lattice.

        sorted_lists = zip(self.NODES["node"], self.NODES["nodxy"])
        sorted_lists = sorted(sorted_lists, key=lambda x: x[0])
        self.NODES["node"], self.NODES["nodxy"] = zip(*sorted_lists) 
        
        self.fields_compiler(constraintField,"nodtag")   
        self.fields_compiler(forceField,"nodval")   
        self.fields_compiler(dispField,"nodisp")    
        
        self.NODES_COMPILED=1      
        return self.NODES


    def compile_rods_dict(self):
        """
        Compile the dictionary relative to rods
        """
        # maximum numbers of vertical/horizontal cells. 
        self.I=self.model2D.shape[0]
        self.J=self.model2D.shape[1]

        # loops variables.
        ROWS = 2*self.model2D.shape[0]+1
        COLS = 2*self.model2D.shape[1]+1

        # counters.
        ROD_COUNTER=0
        NODE_COUNTER1=0
        LINE_COUNTER=0
        NODE_COUNTER2=1
        NODE_COUNTER2_prime=0

        # app variables
        app=0
        app_prime=0

        # steps variables.
        STEP1=3*self.J+1  
        STEP2 = 2*self.J
        STEP2_prime=self.J+1                                                                                                         
        
        for n in range(ROWS): 
            for _ in range(COLS-1):                                                                                                                   
                self.RODS["element"].append(ROD_COUNTER)
                if n%2==0: # horizontal beams.
                    self.RODS["elenod"].append([NODE_COUNTER1 + STEP1*LINE_COUNTER,NODE_COUNTER1 + STEP1*LINE_COUNTER+1])  
                    ROD_COUNTER=ROD_COUNTER+1
                    NODE_COUNTER1=NODE_COUNTER1+1  
                if n%2!=0: # horizontal levers.
                    if app<self.J:
                        self.RODS["elenod"].append([NODE_COUNTER2,NODE_COUNTER2+STEP2-app])
                        NODE_COUNTER2=NODE_COUNTER2+2
                        NODE_COUNTER2_prime=NODE_COUNTER2
                    if app>=self.J:
                        self.RODS["elenod"].append([NODE_COUNTER2_prime,NODE_COUNTER2_prime+STEP2_prime+app_prime])  
                        NODE_COUNTER2_prime=NODE_COUNTER2_prime+1
                        app_prime=app_prime+1            
                    ROD_COUNTER=ROD_COUNTER+1
                    app=app+1
            app=0
            app_prime=0
            if n%2==0:
                NODE_COUNTER1=0
                LINE_COUNTER=LINE_COUNTER+1 
            if n%2!=0:
                NODE_COUNTER2=NODE_COUNTER2+self.J+1

        if self.A.shape[0] == 1 and self.E.shape[0] == 1:
            self.RODS["elefab"] = [self.A] * self.rodsNumber
            self.RODS["elemat"] = [self.E] * self.rodsNumber
        elif self.A.shape[0] == self.rodsNumber and self.E.shape[0] == self.rodsNumber:
            self.RODS["elefab"] = self.A
            self.RODS["elemat"] = self.E
        else: 
            print("ERROR: beam Young Modules and cross sections don't match the number of rods in the lattice.")
        
        self.RODS_COMPILED=1
            
        return self.RODS
    
    def compile_springs_dict(self):
        """
        Compile the dictionary relative to the springs
        """        
        k=0
        for i in range(self.I):
            # app variables
            n=0
            m=0
            p=0
            q=0
            # starting points
            starting_point_low = i*(self.J+self.J*2+1)
            starting_point_high = (i+1)*(self.J+self.J*2+1)
            # per each row, assign lower springs
            for j in range(2*self.J):
                if (j%2) == 0:
                    self.SPRINGS["springnod"].append([starting_point_low+2*n,starting_point_low+2*self.J+m+1])
                    n=n+1
                if (j%2) != 0:
                    self.SPRINGS["springnod"].append([starting_point_low+2*self.J+m+1,starting_point_low+2*n])
                    m=m+1
                self.SPRINGS["spring"].append(k)
                k=k+1
            # per each row, assign upper springs
            for j in range(2*self.J):
                if (j%2) == 0:
                    self.SPRINGS["springnod"].append([starting_point_high+2*p,starting_point_high-self.J+q])
                    p=p+1
                if (j%2) != 0:
                    self.SPRINGS["springnod"].append([starting_point_high-self.J+q,starting_point_high+2*p])
                    q=q+1
                self.SPRINGS["spring"].append(k)
                k=k+1
        
        self.SPRINGS_COMPILED=1

        return self.SPRINGS
        
    def plot(self):
        """
        plot the lattice
        """
        draw_lattice(self.NODES,self.RODS,self.SPRINGS,\
            [self.NODES_COMPILED,self.RODS_COMPILED,self.SPRINGS_COMPILED])
            
    ## END METHODS DEFINITION ##