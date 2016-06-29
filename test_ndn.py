import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nengo
from nengo.dists import Uniform
import nengo_detailed_neurons
from nengo_detailed_neurons.neurons import Bahr2, IntFire1
from nengo_detailed_neurons.synapses import ExpSyn, FixedCurrent

t_sim=2 * np.pi

# Create a 'model' object to which we can add ensembles, connections, etc.  
model = nengo.Network(label="Communications Channel", seed=3145987)
with model:
    # Create an abstract input signal that oscillates as sin(t)
    sin = nengo.Node(lambda x: np.sin(x))

    # Create the neuronal ensembles
    num_A_neurons = 200
    num_B_neurons = 50
    A = nengo.Ensemble(num_A_neurons, dimensions=1, max_rates=Uniform(60, 80))
    B = nengo.Ensemble(num_B_neurons, dimensions=1, neuron_type=Bahr2(), max_rates=Uniform(60, 80))
    C = nengo.Ensemble(num_A_neurons, dimensions=1, max_rates=Uniform(60, 80))
    conn = nengo.Connection(B, C)

    # Connect the input to the first neuronal ensemble
    nengo.Connection(sin, A)

    # Connect the first neuronal ensemble to the second (this is the communication channel)
    solver = nengo.solvers.LstsqL2(True)
    nengo.Connection(A, B, solver=solver, synapse=ExpSyn(0.005))

    sin_probe = nengo.Probe(sin)
    A_probe = nengo.Probe(A, synapse=.01)  # ensemble output 
    B_probe = nengo.Probe(B, synapse=.01)
    C_probe = nengo.Probe(C, synapse=.01)
    A_spikes = nengo.Probe(A.neurons, 'spikes')
    B_spikes = nengo.Probe(B.neurons, 'spikes')
    voltage = nengo.Probe(B.neurons, 'voltage')

sim = nengo.Simulator(model)

n = Bahr2()
gain, bias = n.gain_bias([80], [-0.4])
n.rates([-0.2, 0., 0.2, 0.5, 1.], gain, bias)

sim.run(t_sim)

sns.set(context='poster')
figure, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(sim.trange(), sim.data[sin_probe])
ax1.set_xlabel("t")
ax1.set_ylabel("stimulus")
ax1.set_ylim(-1.2, 1.2)
ax2.set_ylabel("A ({} standard LIF)".format(num_A_neurons))
ax2.plot(sim.trange(), sim.data[A_probe])
ax2.set_xlabel("t")
ax2.set_yticklabels([])
ax2.set_ylim(-1.2, 1.2)
ax3.set_ylabel("B ({} compartmental)".format(num_B_neurons))
ax3.plot(sim.trange(), sim.data[B_probe])
ax3.set_xlabel("t")
ax3.set_yticklabels([])
ax3.set_ylim(-1.2, 1.2)
plt.show()