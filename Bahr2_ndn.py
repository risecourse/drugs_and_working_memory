import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nengo
import nengo_detailed_neurons
import pandas as pd
import os
from nengo.dists import Uniform
from nengo_detailed_neurons.neurons import Bahr2
from nengo_detailed_neurons.synapses import ExpSyn
from nengo.utils.ensemble import tuning_curves
from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool2

def run(exp_params):
    drug_effect_stim={'control':0.0,'PHE':-0.3,'GFC':0.5}
    drug_effect_channel={'control':1.0,'PHE':1.05,'GFC':0.95}

    drug=exp_params[0]
    trial=exp_params[1]
    seed=exp_params[2]
    dt=0.001
    t_delay= 0.1
    t_stim=0.1
    tau = 0.1
    noise_B=0.005
    ramp_scale=0.42
    num_A_neurons = 100
    num_LIF_neurons = 1000
    num_nrn_neurons = 1000
    num_C_neurons = 100
    dt_sample=0.005
    timesteps=np.arange(0,int((t_stim+t_delay)/dt_sample))

    def primary_dataframe(sim,drug,trial,stim_probe,B_LIF_probe,B_nrn_probe,C_LIF_probe,C_nrn_probe):
        columns=('time','drug','trial','stim','ramp','b_lif_stim','b_lif_ramp','b_nrn_stim','b_nrn_ramp','c_lif','c_nrn') 
        df_primary = pd.DataFrame(columns=columns, index=np.arange(0,len(timesteps)))
        i=0
        for t in timesteps:
            stim=sim.data[stim_probe][t][0]
            ramp=sim.data[stim_probe][t][1]
            b_lif_stim=sim.data[B_LIF_probe][t][0]
            b_lif_ramp=sim.data[B_LIF_probe][t][1]
            b_nrn_stim=sim.data[B_nrn_probe][t][0]
            b_nrn_ramp=sim.data[B_nrn_probe][t][1]
            c_lif=sim.data[C_LIF_probe][t][0]
            c_nrn=sim.data[C_nrn_probe][t][0]
            rt=t*dt_sample
            df_primary.loc[i]=[rt,drug,trial,stim,ramp,b_lif_stim,b_lif_ramp,b_nrn_stim,b_nrn_ramp,c_lif,c_nrn]
            i+=1
        return df_primary

    with nengo.Network(seed=seed+trial) as model:
        stim = nengo.Node(lambda t: 1.0*(t<t_stim))
        ramp = nengo.Node(lambda t: 0.4*(t>t_stim))
        noise = nengo.Node(output=np.random.normal(drug_effect_stim[drug],noise_B))

        # Ensembles
        A = nengo.Ensemble(num_A_neurons, dimensions=2)
        B_LIF = nengo.Ensemble(num_LIF_neurons, dimensions=2)
        B_nrn = nengo.Ensemble(num_nrn_neurons, dimensions=2, neuron_type=Bahr2())
        C_LIF = nengo.Ensemble(num_C_neurons, dimensions=1)
        C_nrn = nengo.Ensemble(num_C_neurons, dimensions=1)

        # Connection
        nengo.Connection(stim, A[0])
        nengo.Connection(ramp, A[1])
        solver = nengo.solvers.LstsqL2(True)
        solver2 = nengo.solvers.LstsqL2(True)
        nengo.Connection(A, B_LIF, synapse=tau, transform=tau) #transform=tau for wm
        nengo.Connection(A, B_nrn, solver=solver, synapse=ExpSyn(tau), transform=tau) #transform=tau for wm
        nengo.Connection(B_LIF, B_LIF, synapse=tau)
        nengo.Connection(B_nrn, B_nrn, solver=solver2, synapse=ExpSyn(tau))
        # nengo.Connection(noise,B_LIF.neurons,synapse=tau,transform=np.ones((num_B_neurons,1))*tau)
        # nengo.Connection(noise,B_nrn.neurons,synapse=tau,transform=np.ones((num_B_neurons,1))*tau)
        conn = nengo.Connection(B_LIF[0], C_LIF)
        conn = nengo.Connection(B_nrn[0], C_nrn)

        # Probes
        stim_probe = nengo.Probe(A, synapse=.01)
        B_LIF_probe = nengo.Probe(B_LIF, synapse=.01, sample_every=dt_sample)
        B_nrn_probe = nengo.Probe(B_nrn, synapse=.01, sample_every=dt_sample)
        C_LIF_probe = nengo.Probe(C_LIF, synapse=.01)
        C_nrn_probe = nengo.Probe(C_nrn, synapse=.01)

    # sim = nengo.Simulator(model)
    def reset_channels(drug):
        #strongly enhance the I_h current, by opening HCN channels, to create shunting under control
        for cell in nengo_detailed_neurons.builder.ens_to_cells[B_nrn]:
            cell.neuron.tuft.gbar_ih *= drug_effect_channel[drug]
            cell.neuron.apical.gbar_ih *= drug_effect_channel[drug]
            cell.neuron.recalculate_channel_densities()

    print 'Running drug %s trial %s...' %(drug,trial+1)
    with nengo.Simulator(model,dt=dt) as sim:
        reset_channels(drug)
        sim.run(t_stim+t_delay)
        df_primary=primary_dataframe(sim,drug,trial,stim_probe,B_LIF_probe,B_nrn_probe,C_LIF_probe,C_nrn_probe)
    return df_primary


def id_generator(size=6):
    #http://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python
    import string
    import random
    return ''.join(random.choice(string.ascii_uppercase+string.digits) for _ in range(size))



drugs=['control','PHE','GFC']
seed=3
n_trials=4
n_processes=20
trials=np.arange(n_trials)
pool = Pool2(nodes=n_processes)
exp_params=[]
for drug in drugs:
    for trial in trials:
        exp_params.append([drug, trial, seed])
df_list = pool.map(run, exp_params)
primary_dataframe = pd.concat([df_list[i] for i in range(len(df_list))], ignore_index=True)


print 'Plotting...'
root=os.getcwd()
os.chdir(root+'/data/')
addon=str(id_generator(9))
filename='Bahr2_ndn'
fname=filename+'_'+addon+'.png'
plot_context='poster'
sns.set(context=plot_context)
figure, ((ax2, ax3), (ax4, ax5)) = plt.subplots(2, 2)
# sns.tsplot(time="time",value="stim",data=primary_dataframe,unit="trial",condition='drug',ax=ax1,ci=95)
# sns.tsplot(time="time",value="ramp",data=primary_dataframe,unit="trial",condition='drug',ax=ax1,ci=95)
sns.tsplot(time="time",value="b_lif_stim",data=primary_dataframe,unit="trial",condition='drug',ax=ax2,ci=95)
sns.tsplot(time="time",value="b_lif_ramp",data=primary_dataframe,unit="trial",condition='drug',ax=ax2,ci=95)
sns.tsplot(time="time",value="b_nrn_stim",data=primary_dataframe,unit="trial",condition='drug',ax=ax3,ci=95)
sns.tsplot(time="time",value="b_nrn_ramp",data=primary_dataframe,unit="trial",condition='drug',ax=ax3,ci=95)
sns.tsplot(time="time",value="c_lif",data=primary_dataframe,unit="trial",condition='drug',ax=ax4,ci=95)
sns.tsplot(time="time",value="c_nrn",data=primary_dataframe,unit="trial",condition='drug',ax=ax5,ci=95)
# ax1.set(xlabel='')
ax2.set(xlabel='',ylabel='LIF WM',ylim=(0,0.2))
ax3.set(xlabel='',ylabel='NEURON WM',ylim=(0,0.2))
ax4.set(xlabel='',ylabel='LIF output',ylim=(0,0.1))
ax5.set(xlabel='time (s)',ylabel='NEURON output',ylim=(0,0.1))
figure.savefig(fname)
plt.show()