import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nengo
import pandas as pd
import os
from nengo.dists import Uniform
from nengo.utils.ensemble import tuning_curves
from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool2

def run(exp_params):
    drug_effect_recurrent={'k=1.0':1.0,'k=0.9':0.9,'k=1.1':1.1}

    drug=exp_params[0]
    trial=exp_params[1]
    seed=exp_params[2]
    t_stim=1
    t_delay=8
    dt=0.001
    dt_sample=0.005
    tau = 0.1
    ramp_scale=0.42
    num_A_neurons = 100
    num_LIF_neurons = 100
    num_C_neurons = 100
    noise_B=0.005
    k_neuron=2.3
    timesteps=np.arange(0,int((t_stim+t_delay)/dt_sample))

    def primary_dataframe(sim,drug,trial,C_LIF_probe):
        columns=('time','drug','trial','c_lif') 
        df_primary = pd.DataFrame(columns=columns, index=np.arange(0,len(timesteps)))
        i=0
        for t in timesteps:
            c_lif=sim.data[C_LIF_probe][t][0]
            rt=t*dt_sample
            df_primary.loc[i]=[rt,drug,trial,c_lif]
            i+=1
        return df_primary

    with nengo.Network(seed=seed+trial) as model:
        stim = nengo.Node(lambda t: 1.0*(t<t_stim))
        # ramp = nengo.Node(lambda t: 0.4*(t>t_stim))
        # noise = nengo.Node(output=np.random.normal(drug_effect_stim[drug],noise_B))

        # Ensembles
        A = nengo.Ensemble(num_A_neurons, dimensions=1)
        B_LIF = nengo.Ensemble(num_LIF_neurons, dimensions=1)
        C_LIF = nengo.Ensemble(num_C_neurons, dimensions=1)

        # Connection
        nengo.Connection(stim, A[0])
        nengo.Connection(A, B_LIF, synapse=tau, transform=tau)
        nengo.Connection(B_LIF, B_LIF, synapse=tau, transform=drug_effect_recurrent[drug])
        nengo.Connection(B_LIF[0], C_LIF)

        # Probes
        C_LIF_probe = nengo.Probe(C_LIF, synapse=.01, sample_every=dt_sample)


    print 'Running drug %s trial %s...' %(drug,trial+1)
    with nengo.Simulator(model,dt=dt) as sim:
        sim.run(t_stim+t_delay)
        df_primary=primary_dataframe(sim,drug,trial,C_LIF_probe)
    return df_primary


def id_generator(size=6):
    #http://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python
    import string
    import random
    return ''.join(random.choice(string.ascii_uppercase+string.digits) for _ in range(size))


drugs=['k=1.0','k=0.9','k=1.1']
seed=3
n_trials=10
n_processes=30
trials=np.arange(n_trials)
pool = Pool2(nodes=n_processes)
exp_params=[]
for drug in drugs:
    for trial in trials:
        exp_params.append([drug, trial, seed])

df_list = pool.map(run, exp_params)
primary_dataframe = pd.concat([df for df in df_list], ignore_index=True)
# print primary_dataframe

print 'Plotting...'
root=os.getcwd()
os.chdir(root+'/data/')
addon=str(id_generator(9))
filename='optimal_integrator_multiply_demo'
fname=filename+'_'+addon+'.png'
plot_context='poster'
sns.set(context=plot_context)
figure, (ax1,ax2) = plt.subplots(2, 1)
sns.tsplot(time="time",value="c_lif",data=primary_dataframe,unit="trial",condition='drug',ax=ax2,ci=95)
ax2.set(xlabel='time (s)',ylabel='decoded $\hat{x}(t)$')
figure.savefig(fname)
plt.show()