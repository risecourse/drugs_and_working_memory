# Peter Duggins
# ICCM 2016 Project
# June-August 2016
# Modeling the effects of drugs on working memory

import nengo
from nengo.dists import Choice,Exponential,Uniform
import nengo_gui
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import ipdb

'''Parameters ###############################################'''
#simulation parameters
filename='wm_4'
n_trials=10
dt=0.001 #timestep
dt_sample=0.05 #probe sample_every
t_stim=1.0 #duration of cue presentation
t_delay=8.0 #duration of delay period between cue and decision

decision_type='BG' #which decision procedure to use: 'choice' for noisy choice, 'BG' basal ganglia
drug_type='alpha' #how to simulate the drugs: 'addition','multiply',alpha','NEURON'
drugs=['control','PHE','GFC'] #list of drugs to simulate
drugs_effect={'control':0.00,'PHE':-0.4,'GFC':0.4} #mean of injected stimulus onto wm.neurons
drugs_effect_2={'control':[1.0,1,0],'PHE':[0.5,1.0],'GFC':[2.0,1.0]} #multiplier for alpha/bias in wm
ramp_scale=0.42 #how fast does the 'time' dimension accumulate in WM neurons, default=0.42
BG_delta_ramp=-0.10 #change in ramp_scale when using BG decision procedure, default=-0.1+/-0.01
stim_scale=1.0 #how strong is the stimulus from the visual system
k_multiply=5 #divide drugs_effect by this when calculating the magnitude of the wm recurrent conn
enc_min_cutoff=0.0 #minimum cutoff for "weak" encoders in preferred directions
enc_max_cutoff=0.666 #maximum cutoff for "weak" encoders in preferred directions
sigma_smoothing=0.005 #gaussian smoothing applied to spike data to calculate firing rate
record_spikes=True #should the simulation record and plot firing rates of neurons in WM?
record_spike_fraction=0.01 #number of neurons in WM to add to dataframe and plot
misperceive=0.1 #chance of failing to perceive the cue, causing no info to go into WM
perceived=np.ones(n_trials) #list of correctly percieved (not necessarily remembered) cues
cues=2*np.random.randint(2,size=n_trials)-1 #whether the cues is on the left or right
for n in range(len(perceived)):
	if np.random.rand()<misperceive: perceived[n]=0

#ensemble parameters
neurons_sensory=100 #neurons for the sensory ensemble
neurons_wm=1000 #neurons for workimg memory ensemble
neurons_decide=100 #neurons for decision or basal ganglia
tau_stim=None #synaptic time constant of stimuli to populations
tau=0.01 #synaptic time constant between ensembles
tau_wm=0.1 #synapse on recurrent connection in wm
noise_wm=0.005 #standard deviation of full-spectrum white noise injected into wm.neurons
noise_decision=0.3 #for addition, std of added gaussian noise; 
wm_decay=1.0 #recurrent transform in wm ensemble: set <1.0 for decay

params={
	'filename':filename,
	'n_trials':n_trials,
	'dt':dt,
	'dt_sample':dt_sample,
	't_stim':t_stim,
	't_delay':t_delay,
	'decision_type':decision_type,
	'drug_type':drug_type,
	'drugs':drugs,
	'drugs_effect':drugs_effect,
	'ramp_scale':ramp_scale,
	'BG_delta_ramp':BG_delta_ramp,
	'stim_scale':stim_scale,
	'enc_min_cutoff':enc_min_cutoff,
	'enc_max_cutoff':enc_max_cutoff,
	'sigma_smoothing':sigma_smoothing,
	'record_spikes':record_spikes,
	'record_spike_fraction':record_spike_fraction,
	'misperceive':misperceive,
	# 'cues':cues,
	# 'perceived':perceived,

	'neurons_sensory':neurons_sensory,
	'neurons_wm':neurons_wm,
	'neurons_decide':neurons_decide,
	'tau_stim':tau_stim,
	'tau':tau,
	'tau_wm':tau_wm,
	'noise_wm':noise_wm,
	'noise_decision':noise_decision,
	'wm_decay':wm_decay,
}



'''helper functions ###############################################'''
drug='control'
n=0

def stim_function(t):
	if t < t_stim and perceived[n]!=0: return stim_scale*cues[n]
	else: return 0

def ramp_function(t):
	if t > t_stim: return ramp_scale + BG_delta_ramp * (decision_type=='BG')
	else: return 0

def noise_bias_function(t):
	if drug_type=='addition':
		return np.random.normal(drugs_effect[drug],noise_wm)
	else:
		return np.random.normal(0.0,noise_wm)

def noise_decision_function(t):
	return np.random.normal(0.0,noise_decision)

def wm_recurrent_function(x):
	if drug_type == 'multiply':
		return x * (wm_decay + drugs_effect[drug] / k_multiply)
	else:
		return x * wm_decay

def decision_function(x):
	output=0.0
	if decision_type=='choice':
		value=x[0]+x[1]
		if value > 0.0: output = 1.0
		elif value < 0.0: output = -1.0
	elif decision_type=='BG':
		# print 'x[0]',x[0]
		# print 'x[1]',x[1]
		if x[0] > x[1]: output = 1.0
		elif x[0] < x[1]: output = -1.0
	return output 

def reset_alpha_bias():
	wm.gain = sim.data[wm].gain * drugs_effect_2[drug][0]
	wm.bias = sim.data[wm].bias * drugs_effect_2[drug][1]

def update_dataframe(i):
	for t in timesteps:
		wm_val=np.abs(sim.data[probe_wm][t][0])
		output_val=sim.data[probe_output][t][0]
		correct = get_correct(cues[n],output_val)
		rt=t*dt_sample
		dataframe.loc[i]=[rt,drug,wm_val,output_val,correct,n]
		i+=1
	return i #for propper indexing when appending to the dataframe in the simulation loop

def update_spike_dataframe(j,n):
	firing_rate_array = np.zeros((neurons_wm,len(timesteps)))
	t_width = 0.2
	t_h = np.arange(t_width / dt) * dt - t_width / 2.0
	h = np.exp(-t_h ** 2 / (2 * sigma_smoothing ** 2))
	h = h / np.linalg.norm(h, 1)
	for f in range(int(neurons_wm*record_spike_fraction)):
		enc = sim.data[wm].encoders[f]
		tuning = get_tuning(cues[n],enc)		
		firing_rate = np.convolve(sim.data[probe_spikes][:,f],h,mode='same')
		for t in timesteps:
			rt=t*dt_sample
			spike_dataframe.loc[j]=[rt,drug,f+n*neurons_wm,tuning,firing_rate[t]]
			j+=1
		# print 'appending dataframe for neuron %s' %f
	return j #for propper indexing when appending to the dataframe in the simulation loop

def get_correct(cue,output_val):
	if (cues[n] > 0.0 and output_val > 0.0) or (cues[n] < 0.0 and output_val < 0.0): correct=1
	else: correct=0
	return correct
			
def get_tuning(cue,enc):
	if (cue > 0.0 and enc_min_cutoff < enc[0] < enc_max_cutoff) or \
		(cue < 0.0 and -1.0*enc_max_cutoff < enc[0] < -1.0*enc_min_cutoff): tuning='weak'
	elif (cue > 0.0 and enc[0] > enc_max_cutoff) or \
		(cue < 0.0 and enc[0] < -1.0*enc_max_cutoff): tuning='strong'
	else: tuning='nonpreferred'
	return tuning



'''model definition ###############################################'''
model=nengo.Network(label='Working Memory DRT with Drugs')
with model:

	#Ensembles
	#Inputs
	stim = nengo.Node(output=stim_function)
	ramp = nengo.Node(output=ramp_function)
	sensory = nengo.Ensemble(neurons_sensory,2)
	noise_wm_node = nengo.Node(output=noise_bias_function)
	#Working Memory
	wm = nengo.Ensemble(neurons_wm,2)
	#Decision
	if decision_type=='choice':
		decision = nengo.Ensemble(neurons_decide,2)
		noise_decision_node = nengo.Node(output=noise_decision_function)	
	elif decision_type=='BG':
		utilities = nengo.networks.EnsembleArray(neurons_sensory,n_ensembles=2)
		BG = nengo.networks.BasalGanglia(dimensions=2)
		decision = nengo.networks.EnsembleArray(neurons_decide,n_ensembles=2,
					intercepts=nengo.dists.Uniform(0.2,1),encoders=nengo.dists.Uniform(1,1))
		temp = nengo.Ensemble(neurons_decide,2)
		bias = nengo.Node([1]*2)
	#Output
	output = nengo.Ensemble(neurons_decide,1)

	#Connections
	nengo.Connection(stim,sensory[0],synapse=tau_stim)
	nengo.Connection(ramp,sensory[1],synapse=tau_stim)
	nengo.Connection(sensory,wm,synapse=tau_wm,transform=tau_wm)
	nengo.Connection(wm,wm,synapse=tau_wm,function=wm_recurrent_function)
	nengo.Connection(noise_wm_node,wm.neurons,synapse=tau_wm,transform=np.ones((neurons_wm,1))*tau_wm)
	if decision_type=='choice':	
		nengo.Connection(wm[0],decision[0],synapse=tau) #no ramp information passed
		nengo.Connection(noise_decision_node,decision[1],synapse=None)
		nengo.Connection(decision,output,function=decision_function)
	elif decision_type=='BG':
		nengo.Connection(wm[0],utilities.input,synapse=tau, transform=[[1],[-1]])
		nengo.Connection(bias,decision.input,synapse=tau)
		nengo.Connection(decision.input,decision.output,transform=(np.eye(2)-1),synapse=tau/2.0)
		nengo.Connection(utilities.output,BG.input,synapse=None)
		nengo.Connection(BG.output,decision.input,synapse=tau)
		nengo.Connection(decision.output,temp)
		nengo.Connection(temp,output,function=decision_function)

	#Probes
	probe_wm=nengo.Probe(wm[0],synapse=0.1,sample_every=dt_sample) #no ramp information collected
	probe_spikes=nengo.Probe(wm.neurons, 'spikes', sample_every=dt_sample) #spike data
	probe_output=nengo.Probe(output,synapse=None,sample_every=dt_sample) #decision data



'''simulation and data collection ###############################################'''
#create Pandas dataframe for model data
columns=('time','drug','wm','output','correct','trial')
trials=np.arange(n_trials)
timesteps=np.arange(0,int((t_stim+t_delay)/dt_sample))
dataframe = pd.DataFrame(columns=columns, index=np.arange(0,len(drugs)*len(trials)*len(timesteps)))

# Spiking data storage and smoothing filter
spike_columns=('time','drug','neuron-trial','tuning','firing_rate')
spike_dataframe = pd.DataFrame(columns=spike_columns, index=np.arange(0,len(drugs)*len(trials)*
						len(timesteps)*int(neurons_wm*record_spike_fraction)))

i=0
j=0
for drug in drugs:
    for n in trials:
		print 'Running drug \"%s\", trial %s...' %(drug,n+1)
		sim=nengo.Simulator(model,dt=dt)
		if drug_type == 'alpha': reset_alpha_bias()
		sim.run(t_stim+t_delay)
		i=update_dataframe(i)
		if record_spikes == True: j=update_spike_dataframe(j,n)

#create Pandas dataframe for model data
emp_columns=('time','drug','accuracy','trial')
emp_timesteps = [2.0,4.0,6.0,8.0]
pre_PHE=[0.972, 0.947, 0.913, 0.798]
pre_GFC=[0.970, 0.942, 0.882, 0.766]
post_GFC=[0.966, 0.928, 0.906, 0.838]
post_PHE=[0.972, 0.938, 0.847, 0.666]
emp_dataframe = pd.DataFrame(columns=emp_columns,index=np.arange(0, 12))
i=0
for t in range(len(emp_timesteps)):
	emp_dataframe.loc[i]=[emp_timesteps[t],'control',np.average([pre_GFC[t],pre_PHE[t]]),0]
	emp_dataframe.loc[i+1]=[emp_timesteps[t],'PHE',post_PHE[t],0]
	emp_dataframe.loc[i+2]=[emp_timesteps[t],'GFC',post_GFC[t],0]
	i+=3


'''data analysis, plotting, exporting ###############################################'''
root=os.getcwd()
os.chdir(root+'/data/')
addon=str(np.random.randint(0,100000))
fname=filename+addon

print 'Exporting Data...'
dataframe.to_pickle(fname+'_data.pkl')
spike_dataframe.to_pickle(fname+'_spike_data.pkl')
param_df=pd.DataFrame([params])
param_df.reset_index().to_json(fname+'_params.json',orient='records')


print 'Plotting...'
sns.set(context='poster')
figure, (ax1, ax2) = plt.subplots(2, 1)
sns.tsplot(time="time",value="wm",data=dataframe,unit="trial",condition='drug',ax=ax1,ci=95)
sns.tsplot(time="time",value="correct",data=dataframe,unit="trial",condition='drug',ax=ax2,ci=95)
sns.tsplot(time="time",value="accuracy",data=emp_dataframe,unit='trial',condition='drug',
			interpolate=False,ax=ax2)
ax1.set(xlabel='',ylabel='abs(WM value)')
ax2.set(xlabel='time (s)',xlim=(2.0,8.0),ylabel='accuracy')
figure.savefig(fname+'_plots.png')
plt.show()

if record_spikes == True:
	sns.set(context='poster')
	figure2, (ax3, ax4) = plt.subplots(1, 2)
	sns.tsplot(time="time",value="firing_rate",data=spike_dataframe.query("tuning=='weak'"),
				unit="neuron-trial",condition='drug',ax=ax3,ci=95)
	sns.tsplot(time="time",value="firing_rate",data=spike_dataframe.query("tuning=='nonpreferred'"),
				unit="neuron-trial",condition='drug',ax=ax4,ci=95)
	ax3.set(xlabel='time (s)',xlim=(0.0,8.0),ylim=(0,500),ylabel='Normalized Firing Rate',title='Preferred Direction')
	ax4.set(xlabel='time (s)',xlim=(0.0,8.0),ylim=(0,500),ylabel='',title='Nonpreferred Direction')
	figure2.savefig(fname+'_firing_rate_plots.png')
	plt.show()

os.chdir(root)