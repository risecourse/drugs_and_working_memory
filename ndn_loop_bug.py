import nengo
import nengo_detailed_neurons
from nengo_detailed_neurons.neurons import Bahr2
from nengo_detailed_neurons.synapses import ExpSyn

for n in range(10):
	print 'trial %s' %n
	with nengo.Network(seed=n) as model:
		stim = nengo.Node(output=1.0)
		inputs = nengo.Ensemble(100,1)
		ens = nengo.Ensemble(100,1,neuron_type=Bahr2())
		output = nengo.Ensemble(100,1)

		solver = nengo.solvers.LstsqL2(True)
		nengo.Connection(stim,inputs)
		nengo.Connection(inputs,ens,synapse=ExpSyn(0.1),solver=solver)
		nengo.Connection(ens,output)

	with nengo.Simulator(model) as sim:
		sim.run(0.01)
		sim.close()
