'''
Peter Duggins, Terry Stewart, Xuan Choo, Chris Eliasmith
Effects of Guanfacine and Phenylephrine on a Spiking Neuron Model of Working Memory
June-August 2016
Main Model File
'''

import pytry
import nengo
import numpy as np
import neuron

from nengo.dists import Choice,Exponential,Uniform
from nengo.rc import rc
from nengo.utils.numpy import rmse

import matplotlib.pyplot as plt

from nengolib.signal import s

from bioneuron_oracle import BahlNeuron, prime_sinusoids, equalpower, OracleSolver

class Trial(pytry.NengoTrial):
	def params(self):
		self.param('network for choosing cue location',
			decision_type='default')
		self.param('neuron type of wm neurons',
			neuron_type='LIF')
		self.param('sampling frequence for spikes',
			dt_sample=0.01)
		self.param('cue period length',
			t_cue=1.0)
		self.param('delay period length',
			t_delay=8.0)
		self.param('oracle / estimate training period length',
			t_train=1.0)
		self.param('validation period length (plots only, for testing)',
			t_test=1.0)
		self.param('signal for training oracle decoders',
			signal_train_oracle='sinusoids')
		self.param('signal for training estimate decoders',
			signal_readout_oracle='sinusoids')
		self.param('signal for validating full decoders (plots only, for testing)',
			signal_test_oracle='sinusoids')
		self.param('magnitude of stimulus onto wm.neurons for drug=stim',
			drug_stim=0.0)
		self.param('k_multiply for wm recurrent connection for drug=functional',
			drug_functional=1.0)
		self.param('k_gain, k_bias for drug=gainbias',
			drug_gainbias=[1.0, 1.0])
		self.param('k_multiply for gbar_ih in the bahl.hoc model (open/close HCN)',
			drug_biophysical=1.0)
		self.param('n_neurons in input ensemble',
			n_neurons_input=100)
		self.param('n_neurons in wm enesemble',
			n_neurons_wm=100)
		self.param('n_neurons in decision/BG ensemble',
			n_neurons_decicision=100)
		self.param('number of synapses on wm recurrent conn w/ bioneurons',
			n_syn=1)
		self.param('minimum firing rate for wm neurons',
			min_rate=150)
		self.param('maximum firing rate for wm neurons',
			max_rate=200)
		self.param('dimensions must equal 2, one for cue, one for time',
			dim=2)
		self.param('radius of wm ensemble',
			radius=1)
		self.param('rate of time accumulation',
			k_time=0.4)
		self.param('rate of cue accumulation',
			k_cue=1.0)
		self.param('default synapse',
			tau=0.01)
		self.param('wm recurrent synapse',
			tau_wm=0.1)
		self.param('std of white noise into wm',
			noise_wm=0.005)  # 0.005
		self.param('std of white noise into decision',
			noise_decision=0.3)  # 0.3
		self.param('regularization',
			reg=0.1)
		self.param('extra Johnson - Lindenstrauss dimensions for wm ensemble',
			JL_dims=0)
		self.param('magnitude of JL_decoders',
			JL_dims_mag=1e-4)
		self.param('probability of misperceiving the cue location',
			prob_misperceive=0.0)  # for bookeeping, only p.perceived affects model
		self.param('did the animal accurately perceive the cue location',
			perceived=True)
		self.param('cue location, -1 for left, +1 for right',
			cue=0),
		self.param('directory where plots are saved',
			plot_dir='pytry_data/')

	def model(self, p):

		assert p.dt <= p.dt_sample, 'dt < dt_sample'
		with nengo.Network(seed=p.seed) as network:

			""" Functions """

			def decision_function(x):
				output=0.0
				if p.decision_type=='default':
					if (x[0] + x[1]) > 0.0: output = 1.0
					elif (x[0] + x[1]) < 0.0: output = -1.0
				elif p.decision_type=='basal_ganglia':
					if x[0] > x[1]: output = 1.0
					elif x[0] < x[1]: output = -1.0
				return output 

			def BG_rescale(x): #rescales -1 to 1 into 0.3 to 1, makes 2-dimensional
				pos_x = 0.5 * (x + 1)
				rescaled = 0.4 + 0.6 * pos_x, 0.4 + 0.6 * (1 - pos_x)
				return rescaled

			if p.neuron_type == 'Bioneuron':
				jl_rng = np.random.RandomState(seed=p.seed)
				d_JL = jl_rng.randn(p.n_neurons_wm, p.JL_dims) * p.JL_dims_mag
				d_recurrent_init = np.hstack((np.zeros((p.n_neurons_wm, p.dim)), d_JL))
				d_decision_1_init = np.zeros((p.n_neurons_wm, p.dim/2))
				d_decision_2_init = np.zeros((p.n_neurons_wm, p.dim/2))
				recurrent_solver = OracleSolver(decoders_bio=d_recurrent_init)
				decision_1_solver = OracleSolver(decoders_bio=d_decision_1_init)
				decision_2_solver = OracleSolver(decoders_bio=d_decision_2_init)


			""" Ensembles """

			# cue = nengo.Node(lambda t: np.cos(2 * np.pi * 2 * t),
			# 	label='stim')
			# time = nengo.Node(lambda t: np.cos(2 * np.pi * 1 * t),
			# 	label='stim2')
			cue = nengo.Node(lambda t: p.k_cue * p.cue * (t < p.t_cue) * (p.perceived))
			time = nengo.Node(lambda t: p.k_time * (t > p.t_cue))
			inputs = nengo.Ensemble(p.n_neurons_input,p.dim,
				seed=p.seed)
			# noise_wm_node = nengo.Node(output=noise_bias_function)
			noise_wm_node = nengo.Node(nengo.processes.WhiteNoise(
				dist=nengo.dists.Gaussian(mean=0, std=p.noise_wm)))
			# white noise with mean = 0 and rms = p.noise_decision
			noise_decision_node = nengo.Node(nengo.processes.WhiteNoise(
				dist=nengo.dists.Gaussian(mean=0, std=p.noise_decision)))
			stim_node = nengo.Node(p.drug_stim)
			if p.neuron_type == 'LIF':
				wm = nengo.Ensemble(p.n_neurons_wm, p.dim,
					neuron_type=nengo.LIF(),
					seed=p.seed,
					radius=p.radius,
					max_rates=nengo.dists.Uniform(p.min_rate, p.max_rate))
				solver = nengo.solvers.LstsqL2(reg=0.01)
			elif p.neuron_type == 'Bioneuron':
				wm = nengo.Ensemble(p.n_neurons_wm,
					dimensions=p.dim+p.JL_dims,
					neuron_type=BahlNeuron(),
					# neuron_type=nengo.LIF(),
					seed=p.seed,
					max_rates=nengo.dists.Uniform(p.min_rate, p.max_rate))
			self.wm = wm
			if p.decision_type == 'default':
				decision = nengo.Ensemble(p.n_neurons_decicision, p.dim)
			elif p.decision_type == 'basal_ganglia':
				utilities = nengo.networks.EnsembleArray(
					p.n_neurons_input,
					n_ensembles=p.dim)
				BG = nengo.networks.BasalGanglia(
					p.dim,
					p.n_neurons_decicision)
				decision = nengo.networks.EnsembleArray(
					p.n_neurons_decicision,
					n_ensembles=p.dim,
					intercepts=Uniform(0.2,1),
					encoders=Uniform(1,1))
				temp = nengo.Ensemble(p.n_neurons_decicision, p.dim)
				bias = nengo.Node([1]*p.dim)
			output = nengo.Ensemble(p.n_neurons_decicision, 1)

			""" Connections """

			nengo.Connection(cue, inputs[0], synapse=None)
			nengo.Connection(time, inputs[1], synapse=None)
			if p.neuron_type == 'LIF':
				nengo.Connection(inputs, wm,
					synapse=p.tau_wm,
					transform=p.tau_wm,
					seed=p.seed)
				self.wm_recurrent=nengo.Connection(wm ,wm,
					synapse=p.tau_wm,
					transform=p.drug_functional,
					seed=p.seed)
				# todo: could break bioneurons
				nengo.Connection(noise_wm_node, wm.neurons,
					synapse=p.tau_wm,
					transform=np.ones((p.n_neurons_wm, 1)) * p.tau_wm)
				nengo.Connection(stim_node, wm.neurons,
					synapse=p.tau_wm,  # drug_stim
					transform=np.ones((p.n_neurons_wm, 1)) * p.tau_wm)
			elif p.neuron_type == 'Bioneuron':
				# These connections will be modified during training
				nengo.Connection(inputs, wm[:p.dim],
					synapse=p.tau_wm,
					transform=p.tau_wm,
					n_syn=p.n_syn,
					seed=p.seed,
					weights_bias_conn=True)
				self.wm_recurrent=nengo.Connection(wm, wm,
					synapse=p.tau_wm,
					n_syn=p.n_syn,
					seed=p.seed,
					solver=recurrent_solver)

			if p.decision_type == 'default':	
				if p.neuron_type == 'LIF':
					self.wm_to_decision = nengo.Connection(wm[0], decision[0],
						synapse=p.tau,
						seed=p.seed)
				elif p.neuron_type == 'Bioneuron':
					self.wm_to_decision = nengo.Connection(wm[0], decision[0],
						synapse=p.tau,
						n_syn=p.n_syn,
						solver=decision_solver,
						seed=p.seed)
				nengo.Connection(noise_decision_node, decision[1], synapse=None)
				nengo.Connection(decision, output, function=decision_function)
			elif p.decision_type == 'basal_ganglia':
				if p.neuron_type == 'LIF':
					self.wm_to_decision = nengo.Connection(wm[0], utilities.input,
						synapse=p.tau,
						function=BG_rescale,
						seed=p.seed)
				elif p.neuron_type == 'Bioneuron':
					# can't easily do a 1D-to-2D function with oracle training
					self.wm_to_decision_1 = nengo.Connection(wm[0], utilities.input[0],
						synapse=p.tau,
						n_syn=p.n_syn,
						seed=p.seed,
						solver=decision_1_solver)
					self.wm_to_decision_2 = nengo.Connection(wm[0], utilities.input[1],
						synapse=p.tau,
						n_syn=p.n_syn,
						seed=p.seed,
						solver=decision_2_solver)
				nengo.Connection(utilities.output, BG.input, synapse=None)
				nengo.Connection(BG.output, decision.input, synapse=p.tau)
				nengo.Connection(noise_decision_node, BG.input[0], synapse=None) #added external noise?
				nengo.Connection(noise_decision_node, BG.input[1], synapse=None) #added external noise?
				nengo.Connection(bias, decision.input, synapse=p.tau)
				nengo.Connection(decision.input, decision.output,
					transform=(np.eye(2)-1),
					synapse=p.tau/2.0)
				nengo.Connection(decision.output, temp)
				nengo.Connection(temp, output, function=decision_function)

			""" Probes """

			if p.neuron_type == 'LIF':
				self.probe_wm=nengo.Probe(wm[0],
					synapse=p.tau_wm,
					sample_every=p.dt_sample)
			self.probe_spikes=nengo.Probe(wm.neurons, 'spikes',
				sample_every=p.dt_sample)
			self.probe_output=nengo.Probe(output,
				synapse=None,
				sample_every=p.dt_sample)

		if p.neuron_type == 'Bioneuron':
			print 'Training decoders for recurrent and decision connection...'

			transform_train = 1.0
			transform_test = 1.0
			freq_train = [2, 1]
			freq_test = [2, 1]
			seed_train = [1, 3]
			seed_test = [3, 1]

			jl_rng = np.random.RandomState(seed=p.seed)
			d_JL = jl_rng.randn(p.n_neurons_wm, p.JL_dims) * p.JL_dims_mag
			d_recurrent_init = np.hstack((np.zeros((p.n_neurons_wm, p.dim)), d_JL))
			d_decision_1_init = np.zeros((p.n_neurons_wm, p.dim/2))
			d_decision_2_init = np.zeros((p.n_neurons_wm, p.dim/2))
			d_readout_init = np.hstack((np.zeros((p.n_neurons_wm, p.dim)), d_JL))

			d_recurrent_new, d_decision_1_extra, d_decision_2_extra, d_readout_extra = train_oracle(
				p=p,
				d_recurrent=d_recurrent_init,
				d_decision_1=d_decision_1_init,
				d_decision_2=d_decision_2_init,
				d_readout=d_readout_init,
				d_JL=d_JL,
				w_train=1.0,
				signal=p.signal_train_oracle,
				freq=freq_train,
				seeds=seed_train,
				transform=transform_train,
				t_final=p.t_train,
				plot=False)
			d_recurrent_extra, d_decision_1_new, d_decision_2_new, d_readout_new = train_oracle(
				p=p,
				d_recurrent=d_recurrent_new,
				d_decision_1=d_decision_1_extra,
				d_decision_2=d_decision_2_extra,
				d_readout=d_readout_extra,
				d_JL=d_JL,
				w_train=0.0,
				signal=p.signal_readout_oracle,
				freq=freq_train,
				seeds=seed_train,
				transform=transform_train,
				t_final=p.t_train,
				plot=False)

			# d_recurrent_extra, d_decision_1_extra, d_decision_2_extra, d_readout_extra = train_oracle(
			# 	p=p,
			# 	d_recurrent=d_recurrent_new,
			# 	d_decision_1=d_decision_1_new,
			# 	d_decision_2=d_decision_2_new,
			# 	d_readout=d_readout_new,
			# 	d_JL=d_JL,
			# 	w_train=0.0,
			# 	signal=p.signal_test_oracle,
			# 	freq=freq_test,
			# 	seeds=seed_test,
			# 	transform=transform_test,
			# 	t_final=p.t_test,
			# 	plot=True)

			with network:
				self.d_recurrent = d_recurrent_new
				self.d_decision_1 = d_decision_1_new
				self.d_decision_2 = d_decision_2_new
				self.d_readout = d_readout_new
				self.wm_recurrent.solver.decoders_bio = self.d_recurrent
				self.wm_to_decision_1.solver.decoders_bio = self.d_decision_1
				self.wm_to_decision_2.solver.decoders_bio = self.d_decision_2

		self.network = network

		return network

	def evaluate(self, p, sim, plt):
		""" gainbias drug manipulation """
		if p.neuron_type != 'Bioneuron':
			# Scale the gains and biases
			self.wm.gain = sim.data[self.wm].gain * p.drug_gainbias[0]
			self.wm.bias = sim.data[self.wm].bias * p.drug_gainbias[1]
			# Set the solver of each connection out of wm to a ProxySolver.
			# This prevents the builder from calculating new optimal decoders
			# for the new gain/bias values,
			# which would effectively 'adapt' the network to the drug stimulation
			self.wm_recurrent.solver = ProxySolver(
				sim.model.params[self.wm_recurrent].weights)
			self.wm_to_decision.solver = ProxySolver(
				sim.model.params[self.wm_to_decision].weights)
			# Rebuild the network to affect the gain/bias change
			sim = nengo.Simulator(self.network,
				seed=p.seed, dt=p.dt)  # , progress_bar=False
		else:
			# Apply the HCN channel opening/closing by manipulating g_HCN (gbar_ih in bahl.hoc)
			for nrn in sim.data[self.wm.neurons]:
				for seg in range(nrn.cell.apical.nseg):
					loc = 1.0 * seg / nrn.cell.apical.nseg  # 0.0 to 1.0
					nrn.cell.apical(loc).gbar_ih *= p.drug_biophysical
				for seg in range(nrn.cell.basal.nseg):
					loc = 1.0 * seg / nrn.cell.basal.nseg  # 0.0 to 1.0
					nrn.cell.basal(loc).gbar_ih *= p.drug_biophysical
				for seg in range(nrn.cell.tuft.nseg):
					loc = 1.0 * seg / nrn.cell.tuft.nseg  # 0.0 to 1.0
					nrn.cell.tuft(loc).gbar_ih *= p.drug_biophysical
				# nrn.cell.soma(0.5).gbar_nat *= p.drug_biophysical
			neuron.init()

		print 'Running Trial...'
		sim.run(p.t_cue+p.t_delay)

		if p.neuron_type == 'LIF':
			wm_data = sim.data[self.probe_wm]
		elif p.neuron_type == 'Bioneuron':
			lpf = nengo.Lowpass(p.tau_wm)
			act_bio = lpf.filt(sim.data[self.probe_spikes], dt=p.dt_sample)
			# wm_data = np.dot(act_bio, self.wm_recurrent.solver.decoders_bio)
			wm_data = np.dot(act_bio, self.d_readout)
			# cheaty way
			# oracle_solver = nengo.solvers.LstsqL2(reg=0.01)
			# decoders_bio_new = oracle_solver(self.act_bio, self.target)[0]
			# wm_data = np.dot(act_bio, decoders_bio_new)
			# wm_data = self.target  # plot ideal value to make sure decoding has correct target

		return dict(
			time=np.arange(p.dt, p.t_cue+p.t_delay, p.dt_sample),
			wm=wm_data,
			output=sim.data[self.probe_output],
			spikes=sim.data[self.probe_spikes],
			encoder=sim.data[self.wm].encoders
			)

def train_oracle(
	p,
	d_recurrent,
	d_decision_1,
	d_decision_2,
	d_readout,
	d_JL,
	w_train,
	readout_LIF = 'LIF',
	signal='sinusoids',
	t_final=1.0,
	freq=1,
	seeds=1,
	transform=1,
	plot=False):

	# Nengo Parameters
	pre_neurons = p.n_neurons_input
	bio_neurons = p.n_neurons_wm
	tau = p.tau
	tau_wm = p.tau_wm
	tau_readout = p.tau_wm
	dt = p.dt
	min_rate = p.min_rate
	max_rate = p.max_rate
	radius = 1
	bio_radius = 1
	n_syn = p.n_syn

	pre_seed = p.seed
	bio_seed = p.seed
	conn_seed = p.seed
	network_seed = p.seed
	sim_seed = p.seed
	post_seed = p.seed
	inter_seed = p.seed
	conn2_seed = p.seed

	max_freq = 5
	rms = 0.25
	n_steps = 10

	dim = p.dim
	reg = p.reg
	t_train = p.t_train
	t_test = p.t_test
	cutoff = 0.1
	jl_dims = p.JL_dims
	jl_dim_mag = p.JL_dims_mag

	"""
	Load the recurrent decoders, with the non-JL dimensions,
	scaled by the training factor, w_train. w_train==1 means only oracle
	spikes are fed back to bio, w_train==0 means only bio spikes are fed back,
	and intermediate values are a weighted mix.
	"""
	d_recurrent[:dim] *= (1.0 - w_train)

	def BG_rescale(x): #rescales -1 to 1 into 0.3 to 1, makes 2-dimensional
		pos_x = 0.5 * (x + 1)
		rescaled = 0.4 + 0.6 * pos_x, 0.4 + 0.6 * (1 - pos_x)
		return rescaled

	"""
	Define the network
	"""
	with nengo.Network(seed=network_seed) as network:

		if signal == 'sinusoids':
			stim = nengo.Node(lambda t: np.cos(2 * np.pi * freq[0] * t),
				label='stim')
			stim2 = nengo.Node(lambda t: np.cos(2 * np.pi * freq[1] * t),
				label='stim2')
		elif signal == 'white_noise':
			stim = nengo.Node(nengo.processes.WhiteSignal(
				period=t_final, high=max_freq, rms=rms, seed=seeds[0]),
				label='stim')
			stim2 = nengo.Node(nengo.processes.WhiteSignal(
				period=t_final, high=max_freq, rms=rms, seed=seeds[1]),
				label='stim2')
		elif signal == 'step':
			stim = nengo.Node(lambda t:
				np.linspace(-freq, freq, n_steps)[int((t % t_final)/(t_final/n_steps))])
			stim2 = nengo.Node(lambda t:
				np.linspace(freq, -freq, n_steps)[int((t % t_final)/(t_final/n_steps))])
		elif signal == 'constant':
			stim = nengo.Node(lambda t: freq[0])
			stim2 = nengo.Node(lambda t: freq[1])
		elif signal == 'ramp':
			stim = nengo.Node(lambda t: p.k_cue * p.cue * (t < p.t_cue) * (p.perceived))
			stim2 = nengo.Node(lambda t: 0*p.k_time * (t > p.t_cue))

		pre = nengo.Ensemble(
			n_neurons=pre_neurons,
			dimensions=dim,
			seed=pre_seed,
			neuron_type=nengo.LIF(),
			radius=radius,
			label='pre')
		bio = nengo.Ensemble(
			n_neurons=bio_neurons,
			dimensions=dim+jl_dims,
			seed=bio_seed,
			neuron_type=BahlNeuron(),
			# neuron_type=nengo.LIF(),
			radius=bio_radius,
			max_rates=nengo.dists.Uniform(min_rate, max_rate),
			label='bio')
		inter = nengo.Ensemble(
			n_neurons=bio_neurons,
			dimensions=dim,
			seed=bio_seed,
			neuron_type=nengo.LIF(),
			max_rates=nengo.dists.Uniform(min_rate, max_rate),
			# radius=radius,
			label='inter')
		lif = nengo.Ensemble(
			n_neurons=bio.n_neurons,
			dimensions=dim,
			seed=bio.seed,
			max_rates=nengo.dists.Uniform(min_rate, max_rate),
			# radius=0.1,
			neuron_type=nengo.LIF(),
			label='lif')
		decision = nengo.Node(size_in=dim, label='decision')
		oracle_recurrent = nengo.Node(size_in=dim, label='oracle_recurrent')
		temp_recurrent = nengo.Ensemble(1, dim, neuron_type=nengo.Direct(), label='temp_recurrent')
		oracle_decision = nengo.Node(size_in=dim, label='oracle_decision')
		# oracle_decision = nengo.Ensemble(1, dim, neuron_type=nengo.Direct(), label='oracle_decision')
		temp = nengo.Node(size_in=dim, label='temp')

		recurrent_solver = OracleSolver(decoders_bio = d_recurrent)
		decision_1_solver = OracleSolver(decoders_bio = d_decision_1)
		decision_2_solver = OracleSolver(decoders_bio = d_decision_2)

		nengo.Connection(stim, pre[0],
			synapse=None)
		nengo.Connection(stim2, pre[1],
			synapse=None)
		''' Connect stimuli (spikes) feedforward to non-JL_dims of bio '''
		nengo.Connection(pre, bio[:dim],
			weights_bias_conn=True,
			seed=conn_seed,
			synapse=tau_wm,
			transform=transform*tau_wm)
		nengo.Connection(pre, lif,
			synapse=tau_wm,
			transform=transform*tau_wm)
		''' Connect recurrent (spikes) feedback to all dims of bio '''
		nengo.Connection(bio, bio,
			seed=conn2_seed,
			synapse=tau_wm,
			solver=recurrent_solver)
		nengo.Connection(lif, lif,
			synapse=tau_wm)
		nengo.Connection(stim, oracle_recurrent[0],
			synapse=1/s,
			transform=transform)
		nengo.Connection(stim2, oracle_recurrent[1],
			synapse=1/s,
			transform=transform)
		nengo.Connection(oracle_recurrent, inter,
			seed=conn2_seed,
			synapse=None,
			transform=1)
		nengo.Connection(inter, bio[:dim],
			seed=conn2_seed,
			synapse=tau_wm,
			transform=w_train)
		conn_lif = nengo.Connection(lif, temp,
			synapse=tau_wm,
			solver=nengo.solvers.LstsqL2(reg=reg))
		nengo.Connection(bio[0], decision[0],
			synapse=tau,
			solver=decision_1_solver)
		nengo.Connection(bio[0], decision[1],
			synapse=tau,
			solver=decision_2_solver)
		# temps are direct mode ensembles that bypass functions computed on passthrough nodes
		nengo.Connection(stim, temp_recurrent[0],
			synapse=1/s,
			transform=transform)
		nengo.Connection(stim2, temp_recurrent[1],
			synapse=1/s,
			transform=transform)
		nengo.Connection(temp_recurrent[0], oracle_decision,
			synapse=tau,
			function=BG_rescale)  # 1 to 2 dimensions

		probe_stim = nengo.Probe(stim, synapse=None)
		probe_stim2 = nengo.Probe(stim2, synapse=None)
		probe_lif = nengo.Probe(lif, synapse=tau_readout, solver=nengo.solvers.LstsqL2(reg=reg))
		probe_bio_spikes = nengo.Probe(bio.neurons, 'spikes')
		probe_lif_spikes = nengo.Probe(lif.neurons, 'spikes')
		probe_oracle_recurrent = nengo.Probe(oracle_recurrent, synapse=tau_readout)
		probe_oracle_decision_1 = nengo.Probe(oracle_decision[0], synapse=tau_readout)
		probe_oracle_decision_2 = nengo.Probe(oracle_decision[1], synapse=tau_readout)


	"""
	Simulate the network, collect bioneuron activities and target values,
	and apply the oracle method to calculate readout decoders
	"""
	with nengo.Simulator(network, dt=dt, progress_bar=True, seed=sim_seed) as train_sim:
		train_sim.run(t_final)
	lpf = nengo.Lowpass(tau_readout)
	act_bio = lpf.filt(train_sim.data[probe_bio_spikes], dt=dt)
	act_lif = lpf.filt(train_sim.data[probe_lif_spikes], dt=dt)
	# bio readout is always "oracle" for the oracle method training
	d_recurrent_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, train_sim.data[probe_oracle_recurrent])[0]
	d_decision_1_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, train_sim.data[probe_oracle_decision_1])[0]
	d_decision_2_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, train_sim.data[probe_oracle_decision_2])[0]
	d_readout_new = nengo.solvers.LstsqL2(reg=reg)(act_bio, train_sim.data[probe_oracle_recurrent])[0]
	if jl_dims > 0:
		d_recurrent_new = np.hstack((d_recurrent_new, d_JL))
		d_readout_new = np.hstack((d_readout_new, d_JL))

	# d_recurrent_new = train_sim.data[conn_lif].weights.T
	# d_readout_new = train_sim.data[conn_lif].weights.T

	"""
	Use the old readout decoders to estimate the bioneurons' outputs for plotting
	"""
	x_target = train_sim.data[probe_oracle_recurrent]
	x_decision_1_target = train_sim.data[probe_oracle_decision_1]
	x_decision_2_target = train_sim.data[probe_oracle_decision_2]
	xhat_bio = np.dot(act_bio, d_readout)
	xhat_lif = train_sim.data[probe_lif]
	xhat_decision_1 = np.dot(act_bio, d_decision_1)
	xhat_decision_2 = np.dot(act_bio, d_decision_2)
	rmse_bio_dim_1 = rmse(x_target[:,0], xhat_bio[:,0])
	rmse_lif_dim_1 = rmse(x_target[:,0], xhat_lif[:,0])
	rmse_bio_dim_2 = rmse(x_target[:,1], xhat_bio[:,1])
	rmse_lif_dim_2 = rmse(x_target[:,1], xhat_lif[:,1])
	rmse_decision_1 = rmse(xhat_decision_1[:,0], x_decision_1_target[:,0])
	rmse_decision_2 = rmse(xhat_decision_2[:,0], x_decision_2_target[:,0])

	if plot:
		fig, ax = plt.subplots(1, 1)
		ax.plot(train_sim.trange(), xhat_bio[:,0], label='bio dim 1, rmse=%.5f' % rmse_bio_dim_1)
		ax.plot(train_sim.trange(), xhat_lif[:,0], label='lif dim 1, rmse=%.5f' % rmse_lif_dim_1)
		ax.plot(train_sim.trange(), x_target[:,0], label='oracle dim 1')
		ax.plot(train_sim.trange(), xhat_bio[:,1], label='bio dim 2, rmse=%.5f' % rmse_bio_dim_2)
		ax.plot(train_sim.trange(), xhat_lif[:,1], label='lif dim 2, rmse=%.5f' % rmse_lif_dim_2)
		ax.plot(train_sim.trange(), x_target[:,1], label='oracle dim 2')
		if jl_dims > 0:
			ax.plot(train_sim.trange(), xhat_bio[:,2:], label='jm_dims')
		# ax.plot(train_sim.trange(), xhat_decision_1[:,0],
		# 	label='bio decision 1, rmse=%.5f' % rmse_decision_1)
		# ax.plot(train_sim.trange(), x_decision_1_target[:,0], label='orcale decision 1')
		# ax.plot(train_sim.trange(), xhat_decision_2[:,0],
		# 	label='bio decision 2, rmse=%.5f' % rmse_decision_2)
		# ax.plot(train_sim.trange(), x_decision_2_target[:,0], label='orcale decision 2')
		ax.set(xlabel='time (s)', ylabel ='$\hat{x}(t)$')
		ax.legend()
		fig.savefig('plots/oracle_test.png')
		# assert rmse_bio_dim_1 < cutoff
		# assert rmse_bio_dim_2 < cutoff

	return d_recurrent_new, d_decision_1_new, d_decision_2_new, d_readout_new



class ProxySolver(nengo.solvers.Solver):
	# For the drug_gainbias manipulation, so that these neuron parameters
	# can be reset without updating the decoders
	# This solver just returns the old decoders
	def __init__(self,weights): # feed in old decoders
		super(ProxySolver, self).__init__(weights=False)  # decoders, not weights
		self.my_weights = weights
	def __call__(self,A,Y,rng=None,E=None): #called by the builder
		return self.my_weights.T, dict()	

#don't try to remember old decoders
# if drug_type == 'biophysical': rc.set("decoder_cache", "enabled", "False")
# else: rc.set("decoder_cache", "enabled", "True")
# if drug_type == 'biophysical': sim=reset_gain_bias(
# 		P,model,sim,wm,wm_recurrent,wm_to_decision,drug)