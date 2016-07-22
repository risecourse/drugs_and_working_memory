# Peter Duggins
# ICCM 2016 Project
# June-August 2016
# Modeling the effects of drugs on working memory

def run(params):
	import nengo
	from nengo.dists import Choice,Exponential,Uniform
	from nengo.rc import rc
	import nengo_detailed_neurons
	from nengo_detailed_neurons.neurons import Bahr2, IntFire1
	from nengo_detailed_neurons.synapses import ExpSyn, FixedCurrent
	import numpy as np
	import pandas as pd

	decision_type=params[0]
	drug_type=params[1]
	drug = params[2]
	trial = params[3]
	seed = params[4]
	my_params = params[5]

	dt=my_params['dt']
	dt_sample=my_params['dt_sample']
	t_cue=my_params['t_cue']
	t_delay=my_params['t_delay']
	drug_effect_cue=my_params['drug_effect_cue']
	drug_effect_inhibit=my_params['drug_effect_inhibit']
	drug_effect_recurrent=my_params['drug_effect_recurrent']
	drug_effect_gainbias=my_params['drug_effect_gainbias']
	drug_effect_channel=my_params['drug_effect_channel']
	enc_min_cutoff=my_params['enc_min_cutoff']
	enc_max_cutoff=my_params['enc_max_cutoff']
	sigma_smoothing=my_params['sigma_smoothing']
	frac=my_params['frac']
	neurons_inputs=my_params['neurons_inputs']
	neurons_wm=my_params['neurons_wm']
	neurons_decide=my_params['neurons_decide']
	ramp_scale=my_params['ramp_scale']
	cue_scale=my_params['cue_scale']
	tau=my_params['tau']
	tau_wm=my_params['tau_wm']
	noise_wm=my_params['noise_wm']
	noise_decision=my_params['noise_decision']
	perceived=my_params['perceived']
	cues=my_params['cues']
	timesteps=np.arange(0,int((t_cue+t_delay)/dt_sample))

	'''helper functions ###############################################'''
	'''stimuli and transforms'''
	def cue_function(t):
		if t < t_cue and perceived[trial]!=0:
			return cue_scale * cues[trial]
		else: return 0

	def ramp_function(t):
		if t > t_cue:
			return ramp_scale
		else: return 0

	def inhibit_function(t):
		return drug_effect_inhibit[drug]

	def noise_bias_function(t):
		if drug_type=='additive':
			return np.random.normal(drug_effect_cue[drug],noise_wm)
		else:
			return np.random.normal(0.0,noise_wm)

	def noise_decision_function(t):
		if decision_type == 'choice':
			return np.random.normal(0.0,noise_decision)
		elif decision_type == 'BG':
			return np.random.normal(0.0,noise_decision,size=2)

	def inputs_function(x):
		return x * tau_wm

	def wm_recurrent_function(x):
		if drug_type == 'recurrent':
			return x * drug_effect_recurrent[drug]
		else:
			return x

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

	def BG_rescale(x): #rescales -1 to 1 into 0.3 to 1, makes 2-dimensional
		pos_x = 0.5 * (x + 1)
		rescaled = 0.3 + 0.7 * pos_x, 0.3 + 0.7 * (1 - pos_x)
		return rescaled

	'''drug approximations'''
	if drug_type == 'gainbias': rc.set("decoder_cache", "enabled", "False") #don't try to remember old decoders
	else: rc.set("decoder_cache", "enabled", "True")
	class MySolver(nengo.solvers.Solver):
		#When the simulator builds the network, it looks for a solver to calculate the decoders
		#instead of the normal least-squares solver, we define our own, so that we can return
		#the old decoders
		def __init__(self,weights): #feed in old decoders
			self.weights=False #they are not weights but decoders
			self.my_weights=weights
		def __call__(self,A,Y,rng=None,E=None): #the function that gets called by the builder
			return self.my_weights.T, dict()

	#https://github.com/nengo/nengo/issues/921 - Thanks Terry!
	def parisien_transform(conn, inh_synapse, inh_proportion=0.25): 
	    # only works for ens->ens connections
	    assert isinstance(conn.pre_obj, nengo.Ensemble)
	    assert isinstance(conn.post_obj, nengo.Ensemble)    

	    # make sure the pre and post ensembles have seeds so we can guarantee their params
	    if conn.pre_obj.seed is None: conn.pre_obj.seed = np.random.randint(0x7FFFFFFF)
	    if conn.post_obj.seed is None: conn.post_obj.seed = np.random.randint(0x7FFFFFFF)

	    # compute the encoders, decoders, and tuning curves
	    model2 = nengo.Network(add_to_container=False)
	    model2.ensembles.append(conn.pre_obj)
	    model2.ensembles.append(conn.post_obj)
	    model2.connections.append(conn)
	    sim = nengo.Simulator(model2)
	    enc = sim.data[conn.post_obj].encoders
	    dec = sim.data[conn].weights
	    eval_points = sim.data[conn].eval_points
	    pts, act = nengo.utils.ensemble.tuning_curves(conn.pre_obj, sim, inputs=eval_points)

	    # compute the original weights
	    transform = nengo.utils.builder.full_transform(conn)
	    w = np.dot(enc, np.dot(transform, dec))

	    # compute the bias function, bias encoders, bias decoders, and bias weights
	    total = np.sum(act, axis=1)    
	    bias_d = np.ones(conn.pre_obj.n_neurons) / np.max(total)    
	    bias_func = total / np.max(total)    
	    bias_e = np.max(-w / bias_d, axis=1)
	    bias_w = np.outer(bias_e, bias_d)

	    # add the new model compontents
	    nengo.Connection(conn.pre_obj.neurons, conn.post_obj.neurons, transform=bias_w, synapse=conn.synapse)
	    inh = nengo.Ensemble(n_neurons=int(conn.pre_obj.n_neurons*inh_proportion), radius=conn.pre_obj.radius,
	                    dimensions=1, encoders=nengo.dists.Choice([[1]]))
	    nengo.Connection(conn.pre_obj, inh, solver=nengo.solvers.NnlsL2(),transform=1,
	                    synapse=inh_synapse,**nengo.utils.connection.target_function(pts, bias_func))
	    nengo.Connection(inh, conn.post_obj.neurons, solver=nengo.solvers.NnlsL2(), transform=-bias_e[:,None])

	    return inh #return the inhibitory ensemble for assignment in model		

	def reset_gainbias_bias(model,sim,wm,wm_recurrent,wm_to_decision,drug):
		#set gains and biases as a constant multiple of the old values
		wm.gain = sim.data[wm].gain * drug_effect_gainbias[drug][0]
		wm.bias = sim.data[wm].bias * drug_effect_gainbias[drug][1]
		#set the solver of each of the connections coming out of wm using the custom MySolver class
		#with input equal to the old decoders. We use the old decoders because we don't want the builder
		#to optimize the decoders to the new gainbias/bias values, otherwise it would "adapt" to the drug
		wm_recurrent.solver = MySolver(sim.model.params[wm_recurrent].weights)
		wm_to_decision.solver=MySolver(sim.model.params[wm_to_decision].weights)
		#rebuild the network to affect the gain/bias change	
		sim=nengo.Simulator(model,dt=dt)
		return sim

	def reset_channels(drug):
		#strongly enhance the I_h current, by opening HCN channels, to create shunting under control
		for cell in nengo_detailed_neurons.builder.ens_to_cells[wm]:
		    cell.neuron.tuft.gbar_ih *= drug_effect_channel[drug]
		    cell.neuron.apical.gbar_ih *= drug_effect_channel[drug]
		    cell.neuron.recalculate_channel_densities()

	'''dataframe initialization'''
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
		cue = nengo.Node(output=cue_function)
		ramp = nengo.Node(output=ramp_function)
		inputs = nengo.Ensemble(neurons_inputs,2)
		noise_wm_node = nengo.Node(output=noise_bias_function)
		noise_decision_node = nengo.Node(output=noise_decision_function)
		#Working Memory
		if drug_type == 'NEURON':
			wm = nengo.Ensemble(neurons_wm,2,neuron_type=Bahr2(),max_rates=Uniform(70,140))
		else:
			wm = nengo.Ensemble(neurons_wm,2)
		#Decision
		if decision_type=='choice':
			decision = nengo.Ensemble(neurons_decide,2)
		elif decision_type=='BG':
			utilities = nengo.networks.EnsembleArray(neurons_inputs,n_ensembles=2)
			BG = nengo.networks.BasalGanglia(2,neurons_decide)
			decision = nengo.networks.EnsembleArray(neurons_decide,n_ensembles=2,
						intercepts=Uniform(0.2,1),encoders=Uniform(1,1))
			temp = nengo.Ensemble(neurons_decide,2)
			bias = nengo.Node([1]*2)
		#Output
		output = nengo.Ensemble(neurons_decide,1)

		#Connections
		nengo.Connection(cue,inputs[0],synapse=None)
		nengo.Connection(ramp,inputs[1],synapse=None)
		if drug_type == 'NEURON':
			solver_cue = nengo.solvers.LstsqL2(True)
			solver_wm = nengo.solvers.LstsqL2(True)
			nengo.Connection(inputs,wm,synapse=ExpSyn(tau_wm),function=inputs_function,solver=solver_cue)
			wm_recurrent=nengo.Connection(wm,wm,synapse=ExpSyn(tau_wm),function=wm_recurrent_function,solver=solver_wm)	
			nengo.Connection(noise_wm_node,wm.neurons,synapse=tau_wm,transform=np.ones((neurons_wm,1))*tau_wm)
		elif drug_type == 'inhibit':
			nengo.Connection(inputs,wm,synapse=tau_wm,function=inputs_function)
			wm_recurrent=nengo.Connection(wm,wm,synapse=tau_wm,function=wm_recurrent_function)
			'''uncomment for parisien transform'''
			# wm_inhibit=parisien_transform(wm_recurrent, inh_synapse=tau_wm)
			# stim_inhibit=nengo.Node(output=inhibit_function)
			# nengo.Connection(stim_inhibit,wm_inhibit,synapse=tau_wm)
			nengo.Connection(noise_wm_node,wm.neurons,synapse=tau_wm,transform=np.ones((neurons_wm,1))*tau_wm)						
		else:
			nengo.Connection(inputs,wm,synapse=tau_wm,function=inputs_function)
			wm_recurrent=nengo.Connection(wm,wm,synapse=tau_wm,function=wm_recurrent_function)
			nengo.Connection(noise_wm_node,wm.neurons,synapse=tau_wm,transform=np.ones((neurons_wm,1))*tau_wm)
		if decision_type=='choice':	
			wm_to_decision=nengo.Connection(wm[0],decision[0],synapse=tau)
			nengo.Connection(noise_decision_node,decision[1],synapse=None)
			nengo.Connection(decision,output,function=decision_function)
		elif decision_type=='BG':
			wm_to_decision=nengo.Connection(wm[0],utilities.input,synapse=tau,function=BG_rescale)
			nengo.Connection(utilities.output,BG.input,synapse=None)
			nengo.Connection(BG.output,decision.input,synapse=tau)
			nengo.Connection(noise_decision_node,BG.input,synapse=None) #added external noise?
			nengo.Connection(bias,decision.input,synapse=tau)
			nengo.Connection(decision.input,decision.output,transform=(np.eye(2)-1),synapse=tau/2.0)
			nengo.Connection(decision.output,temp)
			nengo.Connection(temp,output,function=decision_function)

		#Probes
		probe_wm=nengo.Probe(wm[0],synapse=0.01,sample_every=dt_sample)
		probe_spikes=nengo.Probe(wm.neurons, 'spikes', sample_every=dt_sample)
		probe_output=nengo.Probe(output,synapse=None,sample_every=dt_sample)

	'''simulation ###############################################'''
	print 'Running drug \"%s\", trial %s...' %(drug,trial+1)
	with nengo.Simulator(model,dt=dt) as sim:
		if drug_type == 'gainbias': sim=reset_gainbias_bias(model,sim,wm,wm_recurrent,wm_to_decision,drug)
		if drug_type == 'NEURON': reset_channels(drug)
		sim.run(t_cue+t_delay)
		df_primary=primary_dataframe(sim,drug,trial,probe_wm,probe_output)
		df_firing=firing_dataframe(sim,drug,trial,sim.data[wm],probe_spikes)
	return [df_primary, df_firing]


'''random helper functions'''
def id_generator(size=6):
	#http://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python
	import string
	import random
	return ''.join(random.choice(string.ascii_uppercase+string.digits) for _ in range(size))

def import_params(filename):
	import pandas as pd
	the_params=eval(open(filename).read())
	return the_params

def generate_cues(misperceive,n_trials,seed=3):
	import numpy as np
	trials=np.arange(n_trials)
	perceived=np.ones(n_trials) #list of correctly perceived (not necessarily remembered) cues
	rng=np.random.RandomState(seed=seed)
	cues=2*rng.randint(2,size=n_trials)-1 #whether the cues is on the left or right
	for n in range(len(perceived)): 
		if rng.rand()<misperceive: perceived[n]=0
	return cues,perceived,trials




'''Main ###############################################################'''
def main():
	import matplotlib.pyplot as plt
	import seaborn as sns
	import pandas as pd
	import os
	from pathos.multiprocessing import ProcessingPool as Pool
	from pathos.helpers import freeze_support #for Windows
	# import ipdb

	'''Import Parameters from File'''
	all_params=import_params('parameters.txt')
	seed=all_params['seed']
	n_trials=all_params['n_trials']
	n_processes=all_params['n_processes'] #3*n_trials
	filename=str(all_params['filename'])
	drug_type=str(all_params['drug_type'])
	decision_type=str(all_params['decision_type'])
	drugs=all_params['drugs']
	cues,perceived,trials=generate_cues(all_params['misperceive'],n_trials,seed)
	all_params['cues']=cues
	all_params['perceived']=perceived

	'''Multiprocessing ###############################################'''
	print "Running Experiment: drug_type=%s, decision_type=%s, trials=%s..." %(drug_type,decision_type,n_trials)
	freeze_support()
	pool = Pool(nodes=n_processes)
	exp_params=[]
	for drug in drugs:
		for trial in trials:
			exp_params.append([decision_type, drug_type, drug, trial, seed, all_params])
	# df_list=[run(exp_params[0]),run(exp_params[-1])] #for debugging, since poo.map has unhelpful error messages
	df_list = pool.map(run, exp_params)
	primary_dataframe = pd.concat([df_list[i][0] for i in range(len(df_list))], ignore_index=True)
	firing_dataframe = pd.concat([df_list[i][1] for i in range(len(df_list))], ignore_index=True)
	# print primary_dataframe, firing_dataframe

	'''Plot and Export ###############################################'''
	print 'Exporting Data...'
	root=os.getcwd()
	empirical_dataframe=pd.read_pickle('empirical_data') #add '.pkl' onto filename for windows
	os.chdir(root+'/data/')
	addon=str(id_generator(9))
	fname=filename+'_'+decision_type+'_'+drug_type+'_'+addon
	primary_dataframe.to_pickle(fname+'_primary_data.pkl')
	firing_dataframe.to_pickle(fname+'_firing_data.pkl')
	param_df=pd.DataFrame([all_params])
	param_df.reset_index().to_json(fname+'_params.json',orient='records')

	print 'Plotting...'
	plot_context=all_params['plot_context']
	sns.set(context=plot_context)
	figure, (ax1, ax2) = plt.subplots(2, 1)
	sns.tsplot(time="time",value="wm",data=primary_dataframe,unit="trial",condition='drug',ax=ax1,ci=95)
	sns.tsplot(time="time",value="correct",data=primary_dataframe,unit="trial",condition='drug',ax=ax2,ci=95)
	sns.tsplot(time="time",value="accuracy",data=empirical_dataframe,unit='trial',condition='drug',
				interpolate=False,ax=ax2)
	ax1.set(xlabel='',ylabel='decoded $\hat{cue}$',xlim=(0,9.5),ylim=(0,1),
				title="drug_type=%s, decision_type=%s, trials=%s" %(drug_type,decision_type,n_trials))
	ax2.set(xlabel='time (s)',xlim=(0,9.5),ylim=(0.5,1),ylabel='DRT accuracy')
	figure.savefig(fname+'_primary_plots.png')
	# plt.show()

	sns.set(context=plot_context)
	figure2, (ax3, ax4) = plt.subplots(1, 2)
	if len(firing_dataframe.query("tuning=='weak'"))>0:
		sns.tsplot(time="time",value="firing_rate",unit="neuron-trial",condition='drug',ax=ax3,ci=95,
				data=firing_dataframe.query("tuning=='weak'").reset_index())
	if len(firing_dataframe.query("tuning=='nonpreferred'"))>0:
		sns.tsplot(time="time",value="firing_rate",unit="neuron-trial",condition='drug',ax=ax4,ci=95,
				data=firing_dataframe.query("tuning=='nonpreferred'").reset_index())
	ax3.set(xlabel='time (s)',xlim=(0.0,9.5),ylim=(0,250),ylabel='Normalized Firing Rate',title='Preferred Direction')
	ax4.set(xlabel='time (s)',xlim=(0.0,9.5),ylim=(0,250),ylabel='',title='Nonpreferred Direction')
	figure2.savefig(fname+'_firing_plots.png')
	plt.show()

	os.chdir(root)

if __name__=='__main__':
	main()