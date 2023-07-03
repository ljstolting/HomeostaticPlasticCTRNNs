import numpy as np
import matplotlib.pyplot as plt

########could someday add option to cut HP off at the designated boundaries or not###############

def sigmoid(x):
    return 1/(1+np.exp(-x))
def invsigmoid(x):
    return np.log(x/(1-x))

class CTRNN():

    def __init__(self,size,dt,duration,HPgenome,neurongenome,specificpars = np.ones(15)):
        if np.all(HPgenome) == None:
            HPgenome = np.ones(2*size+3)
            HPgenome[0:size] = 0
        self.Size = size                                         # number of neurons in the circuit
        self.States = np.ones(size)                              # state of the neurons
        self.Outputs = np.zeros(size)                            # neuron outputs
        self.lbs = np.array(HPgenome[0:size])                    # lower bounds of the homeostatic target range
        self.ubs = np.array(HPgenome[size:2*size])               # upper bounds of the homeostatic target range
        self.rhos = np.zeros(size)                               # store the plastic facilitation parameter of each neuron
        self.dt = dt                                             # size of integration timestep, in seconds
        self.duration = duration                                 # duration in seconds
        self.time = np.arange(0.0,self.duration,dt)              # timeseries values in seconds
        self.ctrnn_record = np.zeros((size,len(self.time)))      # place to store data of the node outputs over time
        self.Stepnum = 0                                         # initialize the step count at 0
        self.invadaptWTimeConst = 1/HPgenome[2*size]               # weight adaptive time constant 
        self.invadaptBTimeConst = 1/HPgenome[(2*size)+1]           # bias adaptive time constant 
        self.slidingwindow = int(HPgenome[-1])                   # how far back to go in timesteps to calculate avg_firingrate (& avg_speed)
        self.max_firingrate = np.zeros(self.Size)                # keep track of the maximum firing rate (for diagnostic)
        self.min_firingrate = np.ones(self.Size)                 # keep track of the minimum firing rate (for diagnostic)
        self.Weights = np.reshape(neurongenome[size*2:],(size,size))       # weight matrix (numpy array)
        self.Biases = neurongenome[size:size*2]       # bias values (numpy array)
        self.invTimeConstants = 1.0/neurongenome[0:size]         # inverse taus (numpy array)
        self.bias_record = np.zeros((size,len(self.time)))       # since parameters of the system are changing under the HP, track biases
        self.weight_record = np.zeros((size,size,len(self.time)))# track weights
        self.Inputs = np.zeros((size))                           # external input default to zero
        self.specificpars = specificpars                         # setting where you can give a list of booleans for which pars HP is allowed to change

    def resetStepcount(self):
        self.Stepnum = 0
        
    def setInputs(self,inputs): #external input to each neuron
        self.Inputs = inputs
    
    def setWeights(self, weights): #weight of connection for each neuron pair, going from row (i) to column (j)
        self.Weights = weights
        
    def randomizeWeights(self):
        self.Weights = np.random.uniform(-16,16,size=(self.Size,self.Size))

    def setBiases(self, biases): #bias shift for each neuron
        self.Biases =  biases
    
    def randomizeBiases(self):
        self.Biases = np.random.uniform(-16,16,size=(self.Size))

    def setTimeConstants(self, timeconstants): #time constant for each neuron
        self.TimeConstants =  np.copy(timeconstants)
        self.invTimeConstants = 1.0/self.TimeConstants
        
    def randomizeTimeConstants(self):
        self.TimeConstants = np.random.uniform(0.5,10,size=(self.Size))
        self.invTimeConstants = 1.0/self.TimeConstants
        
    def setAdaptiveTimeConstants(self, adaptiveWtimeconstants, adaptiveBtimeconstants): #time constants for the adaptation of the weights & biases
        self.invadaptWTimeConsts = 1.0/adaptiveWtimeconstants
        self.invadaptBTimeConsts = 1.0/adaptiveBtimeconstants
        
    def randomizeAdaptiveTimeConstants(self):
        self.invadaptWTimeConsts = 1.0/np.random.uniform(10,50,self.Size)
        self.invadaptBTimeConsts = 1.0/np.random.uniform(10,50,self.Size)

    def initializeState(self, s):
        self.States = np.copy(s)
        self.Outputs = sigmoid(self.States+self.Biases)

    def initializeOutput(self,o):
        self.Outputs = np.copy(o)
        self.States = invsigmoid(o) - self.Biases
        
    def plasticFacilitationCalc(self): #calculate and update the value of rho for each neuron, using the mean firing rate from the preceding segment of runtime
        for i in range(self.Size):
            if self.Stepnum < self.slidingwindow:
                self.rhos[i] = 0   #not yet enough data to evaluate average firing rate (don't want to use instantaneous firing rate bc it oscillates)
            else:
                avg_firingrate = np.mean(self.ctrnn_record[i,self.Stepnum-self.slidingwindow:self.Stepnum])
                if avg_firingrate > self.max_firingrate[i]:
                    self.max_firingrate[i] = avg_firingrate
                if avg_firingrate < self.min_firingrate[i]:
                    self.min_firingrate[i] = avg_firingrate
                if  avg_firingrate < self.lbs[i]:
                    self.rhos[i] = 1-(avg_firingrate/self.lbs[i])
                elif avg_firingrate > self.ubs[i]:
                    self.rhos[i] = (self.ubs[i]-avg_firingrate)/(1-self.ubs[i])
                else:
                    self.rhos[i] = 0  #if in range, no change 
            
    
    def updateBiases(self): #use the value of rho for each neuron to dynamically change biases, scaling the change by 1/speed of walking
        for i in range(self.Size):
            if self.specificpars[((self.Size**2)+i)]:
                self.Biases[i] += self.dt * self.invadaptBTimeConst * self.rhos[i]
                if self.Biases[i] >= 16:      #if it goes outside the [-16,16] range, bring it back inside
                    self.Biases[i] = 16       #turned off for these experiments
                if self.Biases[i] <= -16:
                    self.Biases[i] = -16
    
    def updateWeights(self): #use the value of rho for each neuron to dynamically change all incoming weights to that neuron, scaling the change by 1/speed of walking
        specificweights = np.reshape(self.specificpars[:(self.Size**2)],(self.Size,self.Size))
        for j in range(self.Size):
            HPaccess = specificweights[:,j]
            incomingWeights = np.copy(self.Weights[:,j])
            #print("incomingweights to ",j, ": " ,incomingWeights)
            incomingWeights += self.dt * self.invadaptWTimeConst * self.rhos[j] * np.absolute(incomingWeights) * HPaccess
            self.Weights[:,j] = incomingWeights
            for i in range(self.Size):
                if self.Weights[i,j] >= 16:
                    self.Weights[i,j] = 16
                if self.Weights[i,j] <= -16:
                    self.Weights[i,j] = -16
        
    def ctrnnstep(self,adapt): #use the value of the weights and outputs to change the state of each neuron
        #if adapt = true, then we are implementing the adaptive mechanism
        netinput = self.Inputs + np.dot(self.Weights.T, self.Outputs)
        self.States += self.dt * (self.invTimeConstants*(-self.States+netinput))        
        self.Outputs = sigmoid(self.States+self.Biases)
        self.ctrnn_record[:,self.Stepnum] = self.Outputs
        self.plasticFacilitationCalc() #moved up here for this notebook
        self.bias_record[:,self.Stepnum]=self.Biases
        self.weight_record[:,:,self.Stepnum]=self.Weights
        if adapt == True:
            self.updateBiases()
            self.updateWeights()
        self.Stepnum += 1
        
    def run(self,adapt):
        for i in range(len(self.time)):
            self.ctrnnstep(adapt)

    def runonoff(self):
        for i in range(int(len(self.time)/2)):
            self.ctrnnstep(1)
        for i in range(int(len(self.time)/2)):
            self.ctrnnstep(0)

    def plot(self):
        orange = '#fe7c2b'
        margin = 500
        halftime = int(len(self.time)/2)
        if self.Size == 3:
            labels = ["LP","PY","PD"]
        else:
            labels = range(self.Size)
        plt.rcParams["figure.figsize"] = (10,3)
        for i in range(self.Size):
            lab = str(labels[i])
            plt.plot(self.time[halftime-margin:halftime],self.ctrnn_record[i,halftime-margin:halftime],label=lab,color=orange)
            plt.plot(self.time[halftime:halftime+margin],self.ctrnn_record[i,halftime:halftime+margin],color='k')
        # plt.plot(self.time,.5*np.ones(len(self.time)))
        # plt.title("Neural Activity")
        # plt.xlabel("Time (s)")
        plt.ylabel("Neural Outputs")
        plt.xticks([])
        plt.yticks([])
        # plt.legend()
        plt.savefig('notHPenabledoutputs.png', transparent=True)
        plt.show()
        
    def plotparams(self):
        orange = '#fe7c2b'
        margin = 500
        halftime = int(len(self.time)/2)
        plt.rcParams["figure.figsize"] = (10,3)
        for i in range(self.Size):
            for j in range(self.Size):
                idx = 3*i+j
                lab = r'$w_{%s%s}$'%(i,j)
                plt.plot(self.time[halftime-margin:halftime],self.weight_record[i,j,halftime-margin:halftime],label=r'$w_{%s%s}$'%(i,j),color=orange)
        for i in range(self.Size):
            plt.plot(self.time[halftime-margin:halftime],self.bias_record[i,halftime-margin:halftime],label=r'$\theta_%s$'%i,color=orange)
        for i in range(self.Size):
            for j in range(self.Size):
                idx = 3*i+j
                lab = r'$w_{%s%s}$'%(i,j)
                plt.plot(self.time[halftime:halftime+margin],self.weight_record[i,j,halftime:halftime+margin],color='k')
        for i in range(self.Size):
            plt.plot(self.time[halftime:halftime+margin],self.bias_record[i,halftime:halftime+margin],color = 'k')
        # plt.title("CTRNN Parameters")
        plt.xlabel("Time (s)")
        plt.ylabel("CTRNN Parameters")
        plt.xticks([])
        plt.yticks([])
        # plt.legend()
        plt.savefig('notHPenabledparams.png', transparent=True)
        plt.show()