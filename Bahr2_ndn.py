import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nengo
import nengo_detailed_neurons
from nengo.dists import Uniform
from nengo_detailed_neurons.neurons import Bahr2
from nengo_detailed_neurons.synapses import ExpSyn
from nengo.utils.ensemble import tuning_curves
from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool2

def test_wm(drugs):
    drug_effect_stim={'control':0.0,'PHE':-0.3,'GFC':0.5}
    drug=drugs[0]

    t_delay= 0.2
    t_stim=0.1
    tau = 0.1
    noise_B=0.005
    ramp_scale=0.42
    num_A_neurons = 100
    num_B_neurons = 100
    dt_sample=0.001
    k_synapse=1.0 #multiplier to make the expsyn behave more like the default nengo lowpass synapse

    def ramp_function(t):
        if t > t_stim: return ramp_scale
        else: return 0

    # Create a 'model' object to which we can add ensembles, connections, etc.  
    model = nengo.Network(seed=3145987)
    with model:
        # Create an abstract input signal that is a constant value
        stim = nengo.Node(lambda t: 1.0*(t<t_stim))
        ramp = nengo.Node(lambda t: ramp_scale*(t>t_stim))
        noise = nengo.Node(output=np.random.normal(drug_effect_stim[drug],noise_B))

        # Ensembles
        A = nengo.Ensemble(num_A_neurons, dimensions=2)
        B_LIF = nengo.Ensemble(num_B_neurons, dimensions=2)
        B_nrn = nengo.Ensemble(num_B_neurons, dimensions=2, neuron_type=Bahr2())
        C_LIF = nengo.Ensemble(num_A_neurons, dimensions=1)
        C_nrn = nengo.Ensemble(num_A_neurons, dimensions=1)

        # Connection
        nengo.Connection(stim, A[0])
        nengo.Connection(ramp, A[1])
        solver = nengo.solvers.LstsqL2(True)
        solver2 = nengo.solvers.LstsqL2(True)

        nengo.Connection(A, B_LIF, synapse=tau, transform=tau)
        nengo.Connection(B_LIF, B_LIF, synapse=tau, transform=1.0)
        nengo.Connection(A, B_nrn, solver=solver, synapse=ExpSyn(k_synapse*tau), transform=3.0*tau)
        nengo.Connection(B_nrn, B_nrn, solver=solver2, synapse=ExpSyn(k_synapse*tau), transform=1.2*1.0)

        nengo.Connection(noise,B_LIF.neurons,synapse=tau,transform=np.ones((num_B_neurons,1))*tau)
        nengo.Connection(noise,B_nrn.neurons,synapse=tau,transform=np.ones((num_B_neurons,1))*tau)
        conn = nengo.Connection(B_LIF[0], C_LIF)
        conn = nengo.Connection(B_nrn[0], C_nrn)

        # Probes
        stim_probe = nengo.Probe(stim)
        ramp_probe = nengo.Probe(ramp)
        A_probe = nengo.Probe(A, synapse=.01)  # ensemble output 
        B_LIF_probe = nengo.Probe(B_LIF[0], synapse=.01, sample_every=dt_sample)
        B_nrn_probe = nengo.Probe(B_nrn[0], synapse=.01, sample_every=dt_sample)
        C_LIF_probe = nengo.Probe(C_LIF, synapse=.01)
        C_nrn_probe = nengo.Probe(C_nrn, synapse=.01)

    sim = nengo.Simulator(model)

    #strongly enhance the I_h current, by opening HCN channels, to observe shunting effect
    shunting = 200
    for cell in nengo_detailed_neurons.builder.ens_to_cells[B_nrn]:
        cell.neuron.tuft.gbar_ih *= shunting
        cell.neuron.apical.gbar_ih *= shunting
        cell.neuron.recalculate_channel_densities()

    sim.run(t_stim+t_delay)
    return sim

def test_comm(drugs):
    drug_effect_stim={'control':0.0,'PHE':-0.3,'GFC':0.5}
    drug=drugs[0]

    t_delay= 0.1
    t_stim=0.1
    tau = 0.1
    noise_B=0.005
    ramp_scale=0.42
    num_A_neurons = 100
    num_LIF_neurons = 100
    num_nrn_neurons = 200
    dt_sample=0.001

    # Create a 'model' object to which we can add ensembles, connections, etc.  
    model = nengo.Network(seed=3145987)
    with model:
        # Create an abstract input signal that is a constant value
        stim = nengo.Node(lambda t: 0.5*(t<t_stim))
        ramp = nengo.Node(lambda t: 0.5*(t>t_stim))

        # Ensembles
        A = nengo.Ensemble(num_A_neurons, dimensions=2)
        B_LIF = nengo.Ensemble(num_LIF_neurons, dimensions=2)
        B_nrn = nengo.Ensemble(num_nrn_neurons, dimensions=2, neuron_type=Bahr2())
        C_LIF = nengo.Ensemble(num_A_neurons, dimensions=1)
        C_nrn = nengo.Ensemble(num_A_neurons, dimensions=1)

        # Connection
        nengo.Connection(stim, A[0])
        nengo.Connection(ramp, A[1])
        solver = nengo.solvers.LstsqL2(True)

        nengo.Connection(A, B_LIF, synapse=tau)
        nengo.Connection(A, B_nrn, solver=solver, synapse=ExpSyn(tau/2))

        conn = nengo.Connection(B_LIF[0], C_LIF)
        conn = nengo.Connection(B_nrn[0], C_nrn)

        # Probes
        stim_probe = nengo.Probe(A, synapse=.01)
        B_LIF_probe = nengo.Probe(B_LIF, synapse=.01, sample_every=dt_sample)
        B_nrn_probe = nengo.Probe(B_nrn, synapse=.01, sample_every=dt_sample)
        C_LIF_probe = nengo.Probe(C_LIF, synapse=.01)
        C_nrn_probe = nengo.Probe(C_nrn, synapse=.01)

    sim = nengo.Simulator(model)

    # strongly enhance the I_h current, by opening HCN channels, to observe shunting effect
    shunting = 200
    for cell in nengo_detailed_neurons.builder.ens_to_cells[B_nrn]:
        cell.neuron.tuft.gbar_ih *= shunting
        cell.neuron.apical.gbar_ih *= shunting
        cell.neuron.recalculate_channel_densities()

    sim.run(t_stim+t_delay)
    return sim,stim_probe,B_LIF_probe,B_nrn_probe,C_LIF_probe,C_nrn_probe

drugs=[['control']]
sim,stim_probe,B_LIF_probe,B_nrn_probe,C_LIF_probe,C_nrn_probe=test_comm(drugs)

sns.set(context='poster')
figure, ((ax1, ax4), (ax2, ax3)) = plt.subplots(2, 2)
ax1.plot(sim.data[stim_probe],label='cue')
# ax1.plot(sim.data[stim_probe][1],label='ramp')
ax1.set_xlabel("t")
ax1.legend()
ax2.plot(sim.data[B_LIF_probe],label='wm_LIF')
# ax2.plot(sim.data[B_LIF_probe][1],label='wm_LIF ramp')
ax2.set_xlabel("t")
ax2.legend()
ax3.plot(sim.data[B_nrn_probe],label='wm_nrn')
# ax3.plot(sim.data[B_nrn_probe][1],label='wm_nrn ramp')
ax3.set_xlabel("t")
ax3.legend()
ax4.plot(sim.data[C_LIF_probe],label='output_LIF')
ax4.plot(sim.data[C_nrn_probe],label='output_nrn')
ax4.set_xlabel("t")
ax4.legend()
figure.savefig('data/Bahr2_ndn.png')
plt.show()

# sns.set(context='poster')
# figure2, ax1 = plt.subplots(1, 1)
# ax1.plot(*tuning_curves(B, sim))
# ax1.set_xlabel("x")
# ax1.set_ylabel("Interpolated firing rate (1/s)")
# plt.show()