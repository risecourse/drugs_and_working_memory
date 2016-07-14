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
dt=0.001 #timestep
dt_sample=0.05 #probe sample_every
t_stim=1.0 #duration of cue presentation
t_delay=8.0 #duration of delay period between cue and decision
timesteps=np.arange(0,int((t_stim+t_delay)/dt_sample))

drug='control'
decision_type='choice' #which decision procedure to use: 'choice' for noisy choice, 'BG' basal ganglia
drug_type='addition' #how to simulate the drugs: 'addition','multiply',alpha','NEURON',

drug_effect_stim={'control':0.0,'PHE':-0.3,'GFC':0.5,'no_ramp':0.0} #mean of injected stimulus onto wm.neurons
drug_effect_multiply={'control':0.0,'PHE':-0.025,'GFC':0.025} #mean of injected stimulus onto wm.neurons
drug_effect_gain={'control':[1.0,1,0],'PHE':[0.99,1.02],'GFC':[1.05,0.95]} #multiplier for alpha/bias in wm
drug_effect_channel={'control':200.0,'PHE':230,'GFC':160} #multiplier for channel conductances in NEURON cells

k_neuron_sensory=1.0
k_neuron_recur=1.0
delta_rate=0.00 #increase the maximum firing rate of wm neurons for NEURON

enc_min_cutoff=0.3 #minimum cutoff for "weak" encoders in preferred directions
enc_max_cutoff=0.6 #maximum cutoff for "weak" encoders in preferred directions
sigma_smoothing=0.005 #gaussian smoothing applied to spike data to calculate firing rate
frac=0.05 #fraction of neurons in WM to add to dataframe and plot

neurons_sensory=200 #neurons for the sensory ensemble
neurons_wm=100 #neurons for workimg memory ensemble
neurons_decide=100 #neurons for decision or basal ganglia
ramp_scale=0.42 #how fast does the 'time' dimension accumulate in WM neurons, default=0.42
stim_scale=1.0 #how strong is the stimulus from the visual system
tau_stim=None #synaptic time constant of stimuli to populations
tau=0.01 #synaptic time constant between ensembles
tau_wm=0.1 #synapse on recurrent connection in wm
noise_wm=0.005 #standard deviation of full-spectrum white noise injected into wm.neurons
noise_decision=0.3 #for addition, std of added gaussian noise; 
wm_decay=1.0 #recurrent transform in wm ensemble: set <1.0 for decay

misperceive=0.1 #chance of failing to perceive the cue, causing no info to go into WM
perceived=np.ones(1) #list of correctly percieved (not necessarily remembered) cues
cues=2*np.random.randint(2,size=1)-1 #whether the cues is on the left or right
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

def stim_function(t):
	if t < t_stim and perceived[trial]!=0:
		return stim_scale * cues[trial]
	else: return 0

def ramp_function(t):
	if drug=='no_ramp': return 0
	elif drug_type == 'NEURON' and drug_type == 'NEURON':
		return ramp_scale * k_neuron_sensory
	elif t > t_stim:
		return ramp_scale
	else: return 0

def noise_bias_function(t):
	if drug_type=='addition':
		return np.random.normal(drug_effect_stim[drug],noise_wm)
	else:
		return np.random.normal(0.0,noise_wm)

def noise_decision_function(t):
	if decision_type == 'choice':
		return np.random.normal(0.0,noise_decision)
	elif decision_type == 'BG':
		return np.random.normal(0.0,noise_decision,size=2)

def sensory_function(x):
	if drug_type == 'NEURON':
		return x * tau_wm * k_neuron_sensory
	else:
		return x * tau_wm

def wm_recurrent_function(x):
	if drug_type == 'multiply':
		return x * (wm_decay + drug_effect_multiply[drug])
	elif drug_type=='NEURON':
		return x * wm_decay * k_neuron_recur
	else:
		return x * wm_decay

def decision_function(x):
	output=0.0
	if decision_type=='choice':
		value=x[0]+x[1]
		if value > 0.0: output = 1.0
		elif value < 0.0: output = -1.0
	elif decision_type=='BG':
		if x[0] > x[1]: output = 1.0
		elif x[0] < x[1]: output = -1.0
	return output 

def BG_rescale(x): #rescales -1 to 1 into 0.3 to 1
	pos_x = 2 * x + 1
	rescaled = 0.3 + 0.7 * pos_x, 0.3 + 0.7 * (1 - pos_x)
	return rescaled

def reset_alpha_bias(model,sim,wm_recurrent,wm_choice,wm_BG,drug):
	#set gains and biases as a constant multiple of the old values
	wm.gain = sim.data[wm].gain * drug_effect_gain[drug][0]
	wm.bias = sim.data[wm].bias * drug_effect_gain[drug][1]
	#set the solver of each of the connections coming out of wm using the custom MySolver class
	#with input equal to the old decoders. We use the old decoders because we don't want the builder
	#to optimize the decoders to the new alpha/bias values, otherwise it would "adapt" to the drug
	wm_recurrent.solver = MySolver(sim.model.params[wm_recurrent].weights)
	if wm_choice is not None:
		wm_choice.solver = MySolver(sim.model.params[wm_choice].weights)
	if wm_BG is not None:
		#weights[0]=weights[1] for connection from wm to utilities (EnsembleArray.input with n_ens=2)
		wm_BG.solver = MySolver(sim.model.params[wm_BG].weights[0]) 
	#rebuild the network to affect the gain/bias change	
	sim=nengo.Simulator(model,dt=dt)
	return sim

def reset_channels(drug):
	#strongly enhance the I_h current, by opening HCN channels, to create shunting under control
	for cell in nengo_detailed_neurons.builder.ens_to_cells[wm]:
	    cell.neuron.tuft.gbar_ih *= drug_effect_channel[drug]
	    cell.neuron.apical.gbar_ih *= drug_effect_channel[drug]
	    cell.neuron.recalculate_channel_densities()

'''dataframe initialization ###############################################'''

def primary_dataframe(sim,drug,trial,probe_wm,probe_output):
	columns=('time','drug','wm','output','correct','trial') 
	df_primary = pd.DataFrame(columns=columns, index=np.arange(0,len(timesteps)))
	i=0
	for t in timesteps:
		wm_val=np.abs(sim.data[probe_wm][t][0])
		output_val=sim.data[probe_output][t][0]
		correct=get_correct(cues[trial],output_val)
		rt=t*dt_sample
		df_primary.loc[i]=[rt,drug,wm_val,output_val,correct,trial]
		i+=1
	return df_primary

def firing_dataframe(sim,drug,trial,sim_wm,probe_spikes):
	columns=('time','drug','neuron-trial','tuning','firing_rate')
	df_firing = pd.DataFrame(columns=columns, index=np.arange(0,len(timesteps)*int(neurons_wm*frac)))
	t_width = 0.2
	t_h = np.arange(t_width / dt) * dt - t_width / 2.0
	h = np.exp(-t_h ** 2 / (2 * sigma_smoothing ** 2))
	h = h / np.linalg.norm(h, 1)
	j=0
	for nrn in range(int(neurons_wm*frac)):
		enc = sim_wm.encoders[nrn]
		tuning = get_tuning(cues[trial],enc)
		spikes = sim.data[probe_spikes][:,nrn]		
		firing_rate = np.convolve(spikes,h,mode='same')
		for t in timesteps:
			rt=t*dt_sample
			df_firing.loc[j]=[rt,drug,nrn+trial*neurons_wm,tuning,firing_rate[t]]
			j+=1
		# print 'appending dataframe for neuron %s' %f
	return df_firing

def get_correct(cue,output_val):
	if (cue > 0.0 and output_val > 0.0) or (cue < 0.0 and output_val < 0.0): correct=1
	else: correct=0
	return correct
			
def get_tuning(cue,enc):
	if (cue > 0.0 and 0.0 < enc[0] < enc_min_cutoff) or \
		(cue < 0.0 and 0.0 > enc[0] > -1.0*enc_min_cutoff): tuning='superweak'
	if (cue > 0.0 and enc_min_cutoff < enc[0] < enc_max_cutoff) or \
		(cue < 0.0 and -1.0*enc_max_cutoff < enc[0] < -1.0*enc_min_cutoff): tuning='weak'
	elif (cue > 0.0 and enc[0] > enc_max_cutoff) or \
		(cue < 0.0 and enc[0] < -1.0*enc_max_cutoff): tuning='strong'
	else: tuning='nonpreferred'
	return tuning



'''model definition ###############################################'''
with nengo.Network(seed=seed+trial) as model:
    #Ensembles
    #Inputs
    stim = nengo.Node(output=stim_function)
    ramp = nengo.Node(output=ramp_function)
    sensory = nengo.Ensemble(neurons_sensory,2)
    noise_wm_node = nengo.Node(output=noise_bias_function)
    noise_decision_node = nengo.Node(output=noise_decision_function)
    #Working Memory
    if drug_type == 'NEURON':
    	wm = nengo.Ensemble(neurons_wm,2,neuron_type=Bahr2(),max_rates=Uniform(200+delta_rate,400+delta_rate))
    else:
    	wm = nengo.Ensemble(neurons_wm,2)
    #Decision
    if decision_type=='choice':
    	decision = nengo.Ensemble(neurons_decide,2)
    elif decision_type=='BG':
    	utilities = nengo.networks.EnsembleArray(neurons_sensory,n_ensembles=2)
    	BG = nengo.networks.BasalGanglia(2,neurons_decide)
    	decision = nengo.networks.EnsembleArray(neurons_decide,n_ensembles=2,
    				intercepts=Uniform(0.2,1),encoders=Uniform(1,1))
    	temp = nengo.Ensemble(neurons_decide,2)
    	bias = nengo.Node([1]*2)
    #Output
    output = nengo.Ensemble(neurons_decide,1)
    
    #Connections
    nengo.Connection(stim,sensory[0],synapse=tau_stim)
    nengo.Connection(ramp,sensory[1],synapse=tau_stim)
    if drug_type == 'NEURON':
    	solver_stim = nengo.solvers.LstsqL2(True)
    	solver_wm = nengo.solvers.LstsqL2(True)
    	nengo.Connection(sensory,wm,synapse=ExpSyn(tau_wm),function=sensory_function,solver=solver_stim)
    	wm_recurrent=nengo.Connection(wm,wm,synapse=ExpSyn(tau_wm),function=wm_recurrent_function,solver=solver_wm)	
    	nengo.Connection(noise_wm_node,wm.neurons,synapse=tau_wm,transform=np.ones((neurons_wm,1))*tau_wm)							
    else:
    	nengo.Connection(sensory,wm,synapse=tau_wm,function=sensory_function)
    	wm_recurrent=nengo.Connection(wm,wm,synapse=tau_wm,function=wm_recurrent_function)
    	nengo.Connection(noise_wm_node,wm.neurons,synapse=tau_wm,transform=np.ones((neurons_wm,1))*tau_wm)
    wm_choice,wm_BG=None,None
    if decision_type=='choice':	
    	wm_choice=nengo.Connection(wm[0],decision[0],synapse=tau) #no ramp information passed
    	nengo.Connection(noise_decision_node,decision[1],synapse=None)
    	nengo.Connection(decision,output,function=decision_function)
    elif decision_type=='BG':
    	# wm_BG=nengo.Connection(wm[0],utilities.input,synapse=tau,transform=[[1],[-1]])
    	wm_BG=nengo.Connection(wm[0],utilities.input,synapse=tau,function=BG_rescale)
    	nengo.Connection(utilities.output,BG.input,synapse=None)
    	nengo.Connection(BG.output,decision.input,synapse=tau)
    	# nengo.Connection(noise_decision_node,BG.input,synapse=None) #added external noise?
    	nengo.Connection(bias,decision.input,synapse=tau)
    	nengo.Connection(decision.input,decision.output,transform=(np.eye(2)-1),synapse=tau/2.0)
    	nengo.Connection(decision.output,temp)
    	nengo.Connection(temp,output,function=decision_function)

# with nengo.Simulator(model,dt=dt) as sim:
#     if drug_type == 'alpha': sim=reset_alpha_bias(model,sim,wm_recurrent,wm_choice,wm_BG,drug)
#     if drug_type == 'NEURON': reset_channels(drug)
#     sim.run(t_stim+t_delay)