# Peter Duggins
# ICCM 2016 Project
# June-August 2016
# Modeling the effects of drugs on working memory

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
import ipdb
rc.set("decoder_cache", "enabled", "False") #don't try to remember old decoders



'''Parameters ###############################################'''
#simulation parameters
filename='wm_4'
n_trials=1
dt=0.001 #timestep
dt_sample=0.05 #probe sample_every
t_stim=1.0 #duration of cue presentation
t_delay=8.0 #duration of delay period between cue and decision
seed=3 #for the simulator build process, sets tuning curves equal to control before drug application

decision_type='BG' #which decision procedure to use: 'choice' for noisy choice, 'BG' basal ganglia
drug_type='NEURON' #how to simulate the drugs: 'addition','multiply',alpha','NEURON',
drugs=['control']#['control','PHE','GFC'] #list of drugs to simulate; 'no_ramp' (comparison with control)
drug_effect_stim={'control':0.0,'PHE':-0.3,'GFC':0.5,'no_ramp':0.0} #mean of injected stimulus onto wm.neurons
drug_effect_multiply={'control':0.0,'PHE':-0.025,'GFC':0.025} #mean of injected stimulus onto wm.neurons
drug_effect_gain={'control':[1.0,1,0],'PHE':[0.99,1.02],'GFC':[1.05,0.95]} #multiplier for alpha/bias in wm
drug_effect_channel={'control':1.0,'PHE':0.99,'GFC':1.01} #multiplier for channel conductances in NEURON cells
ramp_scale=0.42 #how fast does the 'time' dimension accumulate in WM neurons, default=0.42
BG_delta_ramp=-0.09 #change in ramp_scale when using BG decision procedure, default=-0.1+/-0.01
BG_delta_PHE=0.0 #TODO
stim_scale=1.0 #how strong is the stimulus from the visual system
enc_min_cutoff=0.3 #minimum cutoff for "weak" encoders in preferred directions
enc_max_cutoff=0.6 #maximum cutoff for "weak" encoders in preferred directions
sigma_smoothing=0.005 #gaussian smoothing applied to spike data to calculate firing rate
record_spike_fraction=0.005 #number of neurons in WM to add to dataframe and plot
misperceive=0.1 #chance of failing to perceive the cue, causing no info to go into WM
perceived=np.ones(n_trials) #list of correctly percieved (not necessarily remembered) cues
cues=2*np.random.randint(2,size=n_trials)-1 #whether the cues is on the left or right
for n in range(len(perceived)): 
	if np.random.rand()<misperceive: perceived[n]=0

plot_firing_rate=True #should the simulation record and plot firing rates of neurons in WM?
plot_response_curves=False #plot the response curves? they aren't saved
plot_raster_wm=False #plot wm decoded value with and without the ramp current, with spike raster
plot_context='poster' #seaborn plot context

#ensemble parameters
neurons_sensory=100 #neurons for the sensory ensemble
neurons_wm=100 #neurons for workimg memory ensemble
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
	'drug_effect_stim':drug_effect_stim,
	'drug_effect_multiply':drug_effect_multiply,
	'drug_effect_gain':drug_effect_gain,
	'ramp_scale':ramp_scale,
	'BG_delta_ramp':BG_delta_ramp,
	'stim_scale':stim_scale,
	'enc_min_cutoff':enc_min_cutoff,
	'enc_max_cutoff':enc_max_cutoff,
	'sigma_smoothing':sigma_smoothing,
	'plot_firing_rate':plot_firing_rate,
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
	if t < t_stim and perceived[n]!=0: return stim_scale*cues[n]
	else: return 0

def ramp_function(t):
	if drug=='no_ramp': return 0
	elif t > t_stim: return ramp_scale + BG_delta_ramp * (decision_type=='BG')
	else: return 0

def noise_bias_function(t):
	if drug_type=='addition':
		return np.random.normal(drug_effect_stim[drug],noise_wm)
	else:
		return np.random.normal(0.0,noise_wm)

def noise_decision_function(t):
	return np.random.normal(0.0,noise_decision)

def wm_recurrent_function(x):
	if drug_type == 'multiply':
		return x * (wm_decay + drug_effect_multiply[drug])
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

def reset_alpha_bias(model,sim,wm_recurrent,wm_choice,wm_BG):
	if plot_response_curves == True and n==0:
		sns.set(context='poster')
		figure, (axA, axB) = plt.subplots(2, 1)
		axA.plot(*nengo.utils.ensemble.response_curves(wm,sim))
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
	if plot_response_curves == True and n==0:
		axB.plot(*nengo.utils.ensemble.response_curves(wm,sim))
		axA.set(xlabel='',ylabel='Firing Rate (Hz)',title='control')
		axB.set(xlabel='x along preferred direction',ylabel='Firing Rate (Hz)',title=drug)
		plt.show()
	return sim

def reset_channels():
	for c in nengo_detailed_neurons.builder.ens_to_cells[wm]:
	    c.neuron.soma.gbar_nat *= drug_effect_channel[drug]
	    c.neuron.hillock.gbar_nat *= drug_effect_channel[drug]
	    c.neuron.tuft.gbar_nat *= drug_effect_channel[drug]
	    c.neuron.iseg.gbar_nat *= drug_effect_channel[drug]
	    c.neuron.recalculate_channel_densities()

def update_dataframe(i):
	for t in timesteps:
		wm_val=np.abs(sim.data[probe_wm][t][0])
		output_val=sim.data[probe_output][t][0]
		correct = get_correct(cues[n],output_val)
		rt=t*dt_sample
		dataframe.loc[i]=[rt,drug,wm_val,output_val,correct,n]
		i+=1
	return i #for propper indexing when appending to the dataframe in the simulation loop

def update_firing_rate_dataframe(j,n):
	firing_rate_array = np.zeros((neurons_wm,len(timesteps)))
	t_width = 0.2
	t_h = np.arange(t_width / dt) * dt - t_width / 2.0
	h = np.exp(-t_h ** 2 / (2 * sigma_smoothing ** 2))
	h = h / np.linalg.norm(h, 1)
	for f in range(int(neurons_wm*record_spike_fraction)):
		enc = sim.data[wm].encoders[f]
		tuning = get_tuning(cues[n],enc)
		spikes = sim.data[probe_spikes][:,f]		
		firing_rate = np.convolve(spikes,h,mode='same')
		for t in timesteps:
			rt=t*dt_sample
			firing_rate_dataframe.loc[j]=[rt,drug,f+n*neurons_wm,tuning,firing_rate[t]]
			j+=1
		# print 'appending dataframe for neuron %s' %f
	return j #for propper indexing when appending to the dataframe in the simulation loop

def update_spike_array(k,data):
	spike_array[k]=data
	k+=1 #don't use with more than 1 trial per drug
	return k

def get_correct(cue,output_val):
	if (cues[n] > 0.0 and output_val > 0.0) or (cues[n] < 0.0 and output_val < 0.0): correct=1
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



'''dataframe initialization ###############################################'''
columns=('time','drug','wm','output','correct','trial') 
trials=np.arange(n_trials)
timesteps=np.arange(0,int((t_stim+t_delay)/dt_sample))
dataframe = pd.DataFrame(columns=columns, index=np.arange(0,len(drugs)*len(trials)*len(timesteps)))
firing_rate_columns=('time','drug','neuron-trial','tuning','firing_rate')
firing_rate_dataframe = pd.DataFrame(columns=firing_rate_columns, index=np.arange(0,len(drugs)*len(trials)*
						len(timesteps)*int(neurons_wm*record_spike_fraction)))
emp_columns=('time','drug','accuracy','trial')
emp_timesteps = [2.0,4.0,6.0,8.0]
emp_dataframe = pd.DataFrame(columns=emp_columns,index=np.arange(0, 12))
pre_PHE=[0.972, 0.947, 0.913, 0.798]
pre_GFC=[0.970, 0.942, 0.882, 0.766]
post_GFC=[0.966, 0.928, 0.906, 0.838]
post_PHE=[0.972, 0.938, 0.847, 0.666]
q=0
for t in range(len(emp_timesteps)):
	emp_dataframe.loc[q]=[emp_timesteps[t],'control',np.average([pre_GFC[t],pre_PHE[t]]),0]
	emp_dataframe.loc[q+1]=[emp_timesteps[t],'PHE',post_PHE[t],0]
	emp_dataframe.loc[q+2]=[emp_timesteps[t],'GFC',post_GFC[t],0]
	q+=3
spike_array=np.zeros((2,len(timesteps),neurons_wm))


'''model definition ###############################################'''
i,j,k=0,0,0
print "Experiment: drug_type=%s, decision_type=%s" %(drug_type,decision_type)
for drug in drugs:
    for n in trials:

		with nengo.Network(seed=seed+n) as model:

			#Ensembles
			#Inputs
			stim = nengo.Node(output=stim_function)
			ramp = nengo.Node(output=ramp_function)
			sensory = nengo.Ensemble(neurons_sensory,2)
			noise_wm_node = nengo.Node(output=noise_bias_function)
			#Working Memory
			if drug_type == 'NEURON':
				wm = nengo.Ensemble(neurons_wm,2,neuron_type=Bahr2(),label='wm')
			else:
				wm = nengo.Ensemble(neurons_wm,2,label='wm')
			#Decision
			if decision_type=='choice':
				decision = nengo.Ensemble(neurons_decide,2)
				noise_decision_node = nengo.Node(output=noise_decision_function)	
			elif decision_type=='BG':
				utilities = nengo.networks.EnsembleArray(neurons_sensory,n_ensembles=2)
				BG = nengo.networks.BasalGanglia(dimensions=2)
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
				solver = nengo.solvers.LstsqL2(True)
				nengo.Connection(sensory,wm,synapse=ExpSyn(tau_wm),transform=tau_wm,solver=solver)
				wm_recurrent=nengo.Connection(wm,wm,synapse=ExpSyn(tau_wm),function=wm_recurrent_function,solver=solver)	
				nengo.Connection(noise_wm_node,wm.neurons,synapse=tau_wm,transform=np.ones((neurons_wm,1))*tau_wm)							
			else:
				nengo.Connection(sensory,wm,synapse=tau_wm,transform=tau_wm)
				wm_recurrent=nengo.Connection(wm,wm,synapse=tau_wm,function=wm_recurrent_function)
				nengo.Connection(noise_wm_node,wm.neurons,synapse=tau_wm,transform=np.ones((neurons_wm,1))*tau_wm)
			wm_choice,wm_BG=None,None
			if decision_type=='choice':	
				wm_choice=nengo.Connection(wm[0],decision[0],synapse=tau) #no ramp information passed
				nengo.Connection(noise_decision_node,decision[1],synapse=None)
				nengo.Connection(decision,output,function=decision_function)
			elif decision_type=='BG':
				wm_BG=nengo.Connection(wm[0],utilities.input,synapse=tau,transform=[[1],[-1]])
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



		'''simulation ###############################################'''
		print 'Running drug \"%s\", trial %s...' %(drug,n+1)
		with nengo.Simulator(model,dt=dt) as sim:
			if drug_type == 'alpha': sim=reset_alpha_bias(model,sim,wm_recurrent,wm_choice,wm_BG)
			if drug_type == 'NEURON': reset_channels()
			sim.run(t_stim+t_delay)
			i=update_dataframe(i)
			if plot_firing_rate == True: j=update_firing_rate_dataframe(j,n)
			if plot_raster_wm == True: k=update_spike_array(k,sim.data[probe_spikes])
			sim.close()



'''plot and export ###############################################'''
root=os.getcwd()
os.chdir(root+'/data/')
addon=str(np.random.randint(0,100000))
fname=filename+addon

print 'Exporting Data...'
dataframe.to_pickle(fname+'_data.pkl')
firing_rate_dataframe.to_pickle(fname+'_firing_rate_data.pkl')
param_df=pd.DataFrame([params])
param_df.reset_index().to_json(fname+'_params.json',orient='records')

print 'Plotting...'
sns.set(context=plot_context)
figure, (ax1, ax2) = plt.subplots(2, 1)
sns.tsplot(time="time",value="wm",data=dataframe,unit="trial",condition='drug',ax=ax1,ci=95)
sns.tsplot(time="time",value="correct",data=dataframe,unit="trial",condition='drug',ax=ax2,ci=95)
sns.tsplot(time="time",value="accuracy",data=emp_dataframe,unit='trial',condition='drug',
			interpolate=False,ax=ax2)
ax1.set(xlabel='',ylabel='abs(WM value)',title="drug_type=%s, decision_type=%s" %(drug_type,decision_type))
ax2.set(xlabel='time (s)',xlim=(2.0,8.0),ylabel='accuracy')
figure.savefig(fname+'_plots.png')
plt.show()

if plot_raster_wm==True:
	sns.set(context=plot_context)
	figure3, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2)
	rasterplot(timesteps,spike_array[0],ax=ax5,use_eventplot=True,color='k')
	rasterplot(timesteps,spike_array[-1],ax=ax6,use_eventplot=True,color='k')
	sns.tsplot(time="time",value="wm",unit='trial',ax=ax7,ci=95,
				data=dataframe.query("drug=='no_ramp'"))
	sns.tsplot(time="time",value="wm",unit='trial',ax=ax8,ci=95,
				data=dataframe.query("drug=='control'").reset_index())
	ax5.set(xlabel='',ylabel='neuron \n activity $a_i(t)$')
	ax6.set(xlabel='',ylabel='')
	ax7.set(xlabel='time (s)',ylabel='represented \n value $\hat{x}$')
	ax8.set(xlabel='time (s)',ylabel='')
	figure3.savefig(fname+'_raster_plots.png')
	plt.show()

if plot_firing_rate == True:
	sns.set(context=plot_context)
	figure2, (ax3, ax4) = plt.subplots(1, 2)
	if len(firing_rate_dataframe.query("tuning=='weak'"))>0:
		sns.tsplot(time="time",value="firing_rate",unit="neuron-trial",condition='drug',ax=ax3,ci=95,
				data=firing_rate_dataframe.query("tuning=='weak'").reset_index())
	if len(firing_rate_dataframe.query("tuning=='nonpreferred'"))>0:
		sns.tsplot(time="time",value="firing_rate",unit="neuron-trial",condition='drug',ax=ax4,ci=95,
				data=firing_rate_dataframe.query("tuning=='nonpreferred'").reset_index())
	ax3.set(xlabel='time (s)',xlim=(0.0,8.0),ylim=(0,250),ylabel='Normalized Firing Rate',title='Preferred Direction')
	ax4.set(xlabel='time (s)',xlim=(0.0,8.0),ylim=(0,250),ylabel='',title='Nonpreferred Direction')
	figure2.savefig(fname+'_firing_rate_plots.png')
	plt.show()

os.chdir(root)