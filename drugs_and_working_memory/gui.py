import nengo
from nengo.dists import Choice,Exponential,Uniform
from nengo.utils.matplotlib import rasterplot
from nengo.rc import rc
import nengo_detailed_neurons
from nengo_detailed_neurons.neurons import Bahr2, IntFire1
from nengo_detailed_neurons.synapses import ExpSyn, FixedCurrent
import nengo_gui
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import string
import random
import ipdb

'''Parameters ###############################################'''
#simulation parameters
seed=3 #for the simulator build process, sets tuning curves equal to control before drug application
trial=0
dt=0.001 #P['P['timesteps']s']
dt_sample=0.05 #probe sample_every
t_Cue=1.0 #duration of Cue presentation
t_delay=8.0 #duration of delay period between Cue and Decision
P['P['timesteps']s']s=np.arange(0,int((t_Cue+t_delay)/dt_sample))

drug='control'
Decision_type='default' #which Decision procedure to use: 'default' for noisy choice, 'basal_ganglia' basal ganglia
drug_type='addition' #how to simulate the drugs: 'addition','multiply',alpha','NEURON',

drug_effect_Cue={'control':0.0,'PHE':-0.3,'GFC':0.5,'no_Time':0.0} #mean of injected Cueulus onto WM.neurons
drug_effect_multiply={'control':0.0,'PHE':-0.025,'GFC':0.025} #mean of injected Cueulus onto WM.neurons
drug_effect_gain={'control':[1.0,1,0],'PHE':[0.99,1.02],'GFC':[1.05,0.95]} #multiplier for alpha/bias in WM
drug_effect_channel={'control':200.0,'PHE':230,'GFC':160} #multiplier for channel conductances in NEURON cells

k_neuron_Inputs=1.0
k_neuron_recur=1.0
delta_rate=0.00 #increase the maximum firing rate of WM neurons for NEURON

enc_min_cutoff=0.3 #minimum cutoff for "weak" encoders in preferred directions
enc_max_cutoff=0.6 #maximum cutoff for "weak" encoders in preferred directions
sigma_smoothing=0.005 #gaussian smoothing applied to spike data to calculate firing rate
frac=0.05 #fraction of neurons in WM to add to dataframe and plot

neurons_Inputs=200 #neurons for the Inputs ensemble
neurons_WM=100 #neurons for workimg memory ensemble
neurons_decide=100 #neurons for Decision or basal ganglia
Time_scale=0.42 #how fast does the 'Time' dimension accumulate in WM neurons, default=0.42
Cue_scale=1.0 #how strong is the Cueulus from the visual system
tau_Cue=None #synaptic Time constant of Cueuli to populations
tau=0.01 #synaptic Time constant between ensembles
tau_WM=0.1 #synapse on recurrent connection in WM
Noise_WM=0.005 #standard deviation of full-spectrum white Noise injected into WM.neurons
Noise_Decision=0.3 #for addition, std of added gaussian Noise; 
WM_decay=1.0 #recurrent transform in WM ensemble: set <1.0 for decay

misperceive=0.1 #chance of failing to perceive the Cue, causing no info to go into WM
perceived=np.ones(1) #list of correctly percieved (not necessarily remembered) Cues
Cues=2*np.random.randint(2,size=1)-1 #whether the Cues is on the left or right
for n in range(len(perceived)): 
	if np.random.rand()<misperceive: perceived[n]=0
plot_context='poster' #seaborn plot context


class MySolver(nengo.solvers.Solver):
		#When the simulator builds the network, it looks for a solver to calculate the decoders
		#instead of the normal least-squares solver, we define our own, so that we can return
		#the old decoders
		def __init__(self,weights): #feed in old decoders
			self.weights=False #they are not weights but decoders
			self.my_weights=weights
		def __call__(self,A,Y,rng=None,E=None): #the function that gets called by the builder
			return self.my_weights.T, dict()

def Cue_function(t):
	if t < t_Cue and perceived[trial]!=0:
		return Cue_scale * Cues[trial]
	else: return 0

def Time_function(t):
	if drug=='no_Time': return 0
	elif drug_type == 'NEURON' and drug_type == 'NEURON':
		return Time_scale * k_neuron_Inputs
	elif t > t_Cue:
		return Time_scale
	else: return 0

def Noise_bias_function(t):
	if drug_type=='addition':
		return np.random.normal(drug_effect_Cue[drug],Noise_WM)
	else:
		return np.random.normal(0.0,Noise_WM)

def Noise_Decision_function(t):
	if Decision_type == 'default':
		return np.random.normal(0.0,Noise_Decision)
	elif Decision_type == 'basal_ganglia':
		return np.random.normal(0.0,Noise_Decision,size=2)

def Inputs_function(x):
	if drug_type == 'NEURON':
		return x * tau_WM * k_neuron_Inputs
	else:
		return x * tau_WM

def WM_recurrent_function(x):
	if drug_type == 'multiply':
		return x * (WM_decay + drug_effect_multiply[drug])
	elif drug_type=='NEURON':
		return x * WM_decay * k_neuron_recur
	else:
		return x * WM_decay

def Decision_function(x):
	Output=0.0
	if Decision_type=='default':
		value=x[0]+x[1]
		if value > 0.0: Output = 1.0
		elif value < 0.0: Output = -1.0
	elif Decision_type=='basal_ganglia':
		if x[0] > x[1]: Output = 1.0
		elif x[0] < x[1]: Output = -1.0
	return Output 

def BG_rescale(x): #rescales -1 to 1 into 0.3 to 1
	pos_x = 2 * x + 1
	rescaled = 0.3 + 0.7 * pos_x, 0.3 + 0.7 * (1 - pos_x)
	return rescaled

def reset_alpha_bias(model,sim,WM_recurrent,WM_choice,WM_BG,drug):
	#set gains and biases as a constant multiple of the old values
	WM.gain = sim.data[WM].gain * drug_effect_gain[drug][0]
	WM.bias = sim.data[WM].bias * drug_effect_gain[drug][1]
	#set the solver of each of the connections coming out of WM using the custom MySolver class
	#with input equal to the old decoders. We use the old decoders because we don't want the builder
	#to optimize the decoders to the new alpha/bias values, otherwise it would "adapt" to the drug
	WM_recurrent.solver = MySolver(sim.model.params[WM_recurrent].weights)
	if WM_choice is not None:
		WM_choice.solver = MySolver(sim.model.params[WM_choice].weights)
	if WM_BG is not None:
		WM_BG.solver = MySolver(sim.model.params[WM_BG].weights[0]) 
	#rebuild the network to affect the gain/bias change	
	sim=nengo.Simulator(model,dt=dt)
	return sim

def reset_channels(drug):
	#strongly enhance the I_h current, by opening HCN channels, to create shunting under control
	for cell in nengo_detailed_neurons.builder.ens_to_cells[WM]:
	    cell.neuron.tuft.gbar_ih *= drug_effect_channel[drug]
	    cell.neuron.apical.gbar_ih *= drug_effect_channel[drug]
	    cell.neuron.recalculate_channel_densities()

'''dataframe initialization ###############################################'''

def primary_dataframe(sim,drug,trial,probe_WM,probe_Output):
	columns=('Time','drug','WM','Output','correct','trial') 
	df_primary = pd.DataFrame(columns=columns, index=np.arange(0,len(P['P['timesteps']s']s)))
	i=0
	for t in P['P['timesteps']s']s:
		WM_val=np.abs(sim.data[probe_WM][t][0])
		Output_val=sim.data[probe_Output][t][0]
		correct=get_correct(Cues[trial],Output_val)
		rt=t*dt_sample
		df_primary.loc[i]=[rt,drug,WM_val,Output_val,correct,trial]
		i+=1
	return df_primary

def firing_dataframe(sim,drug,trial,sim_WM,probe_spikes):
	columns=('Time','drug','neuron-trial','tuning','firing_rate')
	df_firing = pd.DataFrame(columns=columns, index=np.arange(0,len(P['P['timesteps']s']s)*int(neurons_WM*frac)))
	t_width = 0.2
	t_h = np.arange(t_width / dt) * dt - t_width / 2.0
	h = np.exp(-t_h ** 2 / (2 * sigma_smoothing ** 2))
	h = h / np.linalg.norm(h, 1)
	j=0
	for nrn in range(int(neurons_WM*frac)):
		enc = sim_WM.encoders[nrn]
		tuning = get_tuning(Cues[trial],enc)
		spikes = sim.data[probe_spikes][:,nrn]		
		firing_rate = np.convolve(spikes,h,mode='same')
		for t in P['P['timesteps']s']s:
			rt=t*dt_sample
			df_firing.loc[j]=[rt,drug,nrn+trial*neurons_WM,tuning,firing_rate[t]]
			j+=1
		# print( 'appending dataframe for neuron %s' %f)
	return df_firing

def get_correct(Cue,Output_val):
	if (Cue > 0.0 and Output_val > 0.0) or (Cue < 0.0 and Output_val < 0.0): correct=1
	else: correct=0
	return correct
			
def get_tuning(Cue,enc):
	if (Cue > 0.0 and 0.0 < enc[0] < enc_min_cutoff) or \
		(Cue < 0.0 and 0.0 > enc[0] > -1.0*enc_min_cutoff): tuning='superweak'
	if (Cue > 0.0 and enc_min_cutoff < enc[0] < enc_max_cutoff) or \
		(Cue < 0.0 and -1.0*enc_max_cutoff < enc[0] < -1.0*enc_min_cutoff): tuning='weak'
	elif (Cue > 0.0 and enc[0] > enc_max_cutoff) or \
		(Cue < 0.0 and enc[0] < -1.0*enc_max_cutoff): tuning='strong'
	else: tuning='nonpreferred'
	return tuning



'''model definition ###############################################'''
with nengo.Network(seed=seed+trial) as model:
    #Ensembles
    #Inputs
    Cue = nengo.Node(output=Cue_function)
    Time = nengo.Node(output=Time_function)
    Inputs = nengo.Ensemble(neurons_Inputs,2)
    Noise_WM = nengo.Node(output=Noise_bias_function)
    Noise_Decision = nengo.Node(output=Noise_Decision_function)
    #Working Memory
    if drug_type == 'NEURON':
    	WM = nengo.Ensemble(neurons_WM,2,neuron_type=Bahr2(),max_rates=Uniform(200+delta_rate,400+delta_rate))
    else:
    	WM = nengo.Ensemble(neurons_WM,2)
    #Decision
    if Decision_type=='default':
    	Decision = nengo.Ensemble(neurons_decide,2)
    elif Decision_type=='basal_ganglia':
    	utilities = nengo.networks.EnsembleArray(neurons_Inputs,n_ensembles=2)
    	BG = nengo.networks.BasalGanglia(2,neurons_decide)
    	Decision = nengo.networks.EnsembleArray(neurons_decide,n_ensembles=2,
    				intercepts=Uniform(0.2,1),encoders=Uniform(1,1))
    	temp = nengo.Ensemble(neurons_decide,2)
    	bias = nengo.Node([1]*2)
    #Output
    Output = nengo.Ensemble(neurons_decide,1)
    
    #Connections
    nengo.Connection(Cue,Inputs[0],synapse=tau_Cue)
    nengo.Connection(Time,Inputs[1],synapse=tau_Cue)
    if drug_type == 'NEURON':
    	solver_Cue = nengo.solvers.LstsqL2(True)
    	solver_WM = nengo.solvers.LstsqL2(True)
    	nengo.Connection(Inputs,WM,synapse=ExpSyn(tau_WM),function=Inputs_function,solver=solver_Cue)
    	WM_recurrent=nengo.Connection(WM,WM,synapse=ExpSyn(tau_WM),function=WM_recurrent_function,solver=solver_WM)	
    	nengo.Connection(Noise_WM,WM.neurons,synapse=tau_WM,transform=np.ones((neurons_WM,1))*tau_WM)							
    else:
    	nengo.Connection(Inputs,WM,synapse=tau_WM,function=Inputs_function)
    	WM_recurrent=nengo.Connection(WM,WM,synapse=tau_WM,function=WM_recurrent_function)
    	nengo.Connection(Noise_WM,WM.neurons,synapse=tau_WM,transform=np.ones((neurons_WM,1))*tau_WM)
    WM_choice,WM_BG=None,None
    if Decision_type=='default':	
    	WM_choice=nengo.Connection(WM[0],Decision[0],synapse=tau) #no Time information passed
    	nengo.Connection(Noise_Decision,Decision[1],synapse=None)
    	nengo.Connection(Decision,Output,function=Decision_function)
    elif Decision_type=='basal_ganglia':
    	# WM_BG=nengo.Connection(WM[0],utilities.input,synapse=tau,transform=[[1],[-1]])
    	WM_BG=nengo.Connection(WM[0],utilities.input,synapse=tau,function=BG_rescale)
    	nengo.Connection(utilities.Output,BG.input,synapse=None)
    	nengo.Connection(BG.Output,Decision.input,synapse=tau)
    	nengo.Connection(Noise_Decision,BG.input,synapse=None) #added external Noise?
    	nengo.Connection(bias,Decision.input,synapse=tau)
    	nengo.Connection(Decision.input,Decision.Output,transform=(np.eye(2)-1),synapse=tau/2.0)
    	nengo.Connection(Decision.Output,temp)
    	nengo.Connection(temp,Output,function=Decision_function)

# with nengo.Simulator(model,dt=dt) as sim:
#     if drug_type == 'alpha': sim=reset_alpha_bias(model,sim,WM_recurrent,WM_choice,WM_BG,drug)
#     if drug_type == 'NEURON': reset_channels(drug)
#     sim.run(t_Cue+t_delay)