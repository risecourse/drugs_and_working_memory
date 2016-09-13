'''
Peter Duggins
Fall 2016
Test nengo_detailed_neurons with communication channel and integrator
'''

def make_df(sim,P,stim_probe,B_LIF_probe,B_NRN_probe,C_LIF_probe,C_NRN_probe):
    import pandas as pd
    import numpy as np
    import ipdb
    timesteps=np.arange(0,int((P['t_stim']+P['t_delay'])/P['dt_sample']))
    columns=('time','drug','trial','a','b_lif','b_NRN','c_lif','c_NRN') 
    df = pd.DataFrame(columns=columns, index=np.arange(0,len(timesteps)))
    i=0
    for t in timesteps:
        rt=t*P['dt_sample']
        a=sim.data[stim_probe][t][0]
        b_lif=sim.data[B_LIF_probe][t][0]
        b_NRN=sim.data[B_NRN_probe][t][0]
        c_lif=sim.data[C_LIF_probe][t][0]
        c_NRN=sim.data[C_NRN_probe][t][0]
        df.loc[i]=[rt,P['drug'],P['trial'],a,b_lif,b_NRN,c_lif,c_NRN]
        i+=1
    return df

def reset_channels(P,ensemble):
    import nengo_detailed_neurons
    for cell in nengo_detailed_neurons.builder.ens_to_cells[ensemble]:
        cell.neuron.tuft.gbar_ih *= P['drug_effect_channel'][P['drug']]
        cell.neuron.apical.gbar_ih *= P['drug_effect_channel'][P['drug']]
        cell.neuron.recalculate_channel_densities()
    return ensemble

def run(P):
    import nengo
    import nengo_detailed_neurons
    from nengo.dists import Uniform
    from nengo_detailed_neurons.neurons import Bahr2
    from nengo_detailed_neurons.synapses import ExpSyn

    with nengo.Network(seed=P['seed']+P['trial']) as model:
        stim = nengo.Node(lambda t: P['cue_scale']*(t<P['t_stim']))
        ramp = nengo.Node(lambda t: P['ramp_scale']*(t>P['t_stim']))

        A = nengo.Ensemble(P['n_neurons_A'], dimensions=2)
        B_LIF = nengo.Ensemble(P['n_neurons_B_LIF'], dimensions=2)
        B_NRN = nengo.Ensemble(P['n_neurons_B_NRN'], dimensions=2, neuron_type=Bahr2(), max_rates=Uniform(60, 80))
        C_LIF = nengo.Ensemble(P['n_neurons_C_LIF'], dimensions=1)
        C_NRN = nengo.Ensemble(P['n_neurons_C_NRN'], dimensions=1)
        # C_NRN = nengo.Ensemble(P['n_neurons_C_NRN'], dimensions=1, neuron_type=Bahr2(), max_rates=Uniform(60, 80))

        nengo.Connection(stim, A[0])
        nengo.Connection(ramp, A[1])
        solver_A_into_B = nengo.solvers.LstsqL2(True)
        # solver_B_into_C = nengo.solvers.LstsqL2(True)
        B_LIF_into_C_LIF = nengo.Connection(B_LIF[0], C_LIF)
        B_NRN_into_C_NRN = nengo.Connection(B_NRN[0], C_NRN)
        # B_NRN_into_C_NRN = nengo.Connection(B_NRN[0], C_NRN, solver=solver_B_into_C, synapse=ExpSyn(P['tau']))

        if P['integrator']==True:
            A_into_B_LIF=nengo.Connection(A, B_LIF,
                                            transform=P['tau_recurrent'], synapse=P['tau_recurrent'])
            A_into_B_NRN=nengo.Connection(A, B_NRN, solver=solver_A_into_B, 
                                            transform=P['tau_recurrent'], synapse=ExpSyn(P['tau_recurrent']))
            # A_into_B_LIF.transform=P['tau_recurrent']
            # A_into_B_LIF.synapse=nengo.Lowpass(tau=P['tau_recurrent'])
            # A_into_B_NRN.transform=P['tau_recurrent']
            # A_into_B_NRN.synapse=nengo.Lowpass(tau=P['tau_recurrent'])
            solver_recurrent_B = nengo.solvers.LstsqL2(True)
            nengo.Connection(B_LIF, B_LIF, synapse=P['tau_recurrent'])
            nengo.Connection(B_NRN, B_NRN, solver=solver_recurrent_B, synapse=ExpSyn(P['tau_recurrent']))
        else:
            A_into_B_LIF=nengo.Connection(A, B_LIF, synapse=P['tau'])
            A_into_B_NRN=nengo.Connection(A, B_NRN, solver=solver_A_into_B, synapse=ExpSyn(P['tau']))

        if P['noise_B']>0.0:
            noise = nengo.Node(output=np.random.normal(P['drug_effect_stim'][P['drug']],P['noise_B']))
            nengo.Connection(noise,B_LIF.neurons,synapse=P['tau'],transform=np.ones((P['num_B_neurons'],1))*P['tau'])
            nengo.Connection(noise,B_NRN.neurons,synapse=P['tau'],transform=np.ones((P['num_B_neurons'],1))*P['tau'])

        a_probe = nengo.Probe(A[0], synapse=0.01, sample_every=P['dt_sample'])
        B_LIF_probe = nengo.Probe(B_LIF[0], synapse=0.01, sample_every=P['dt_sample'])
        B_NRN_probe = nengo.Probe(B_NRN[0], synapse=0.01, sample_every=P['dt_sample'])
        C_LIF_probe = nengo.Probe(C_LIF, synapse=0.01, sample_every=P['dt_sample'])
        C_NRN_probe = nengo.Probe(C_NRN, synapse=0.01, sample_every=P['dt_sample'])

    print 'Running drug %s trial %s...' %(P['drug'],P['trial']+1)
    with nengo.Simulator(model,dt=P['dt']) as sim:
        B_NRN=reset_channels(P,B_NRN)
        sim.run(P['t_stim']+P['t_delay'])
        df=make_df(sim,P,a_probe,B_LIF_probe,B_NRN_probe,C_LIF_probe,C_NRN_probe)
    return df

def init_directory():
    import os
    import string
    import random
    import sys
    root=os.getcwd()
    iden=str(''.join(random.choice(string.ascii_uppercase+string.digits) for _ in range(6)))
    directory=''
    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        directory=root+'/data/'+iden #linux or mac
    elif sys.platform == "win32":
        directory=root+'\\data\\'+iden #windows
    os.makedirs(directory)
    os.chdir(directory)
    # return directory

def export(P,primary_dataframe):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    print 'Plotting and Exporting...'
    sns.set(context='poster')
    figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    sns.tsplot(time="time",value="a",data=primary_dataframe,unit="trial",condition='drug',ax=ax1,ci=95)
    sns.tsplot(time="time",value="a",data=primary_dataframe,unit="trial",condition='drug',ax=ax4,ci=95)
    sns.tsplot(time="time",value="b_lif",data=primary_dataframe,unit="trial",condition='drug',ax=ax2,ci=95)
    sns.tsplot(time="time",value="c_lif",data=primary_dataframe,unit="trial",condition='drug',ax=ax3,ci=95)
    sns.tsplot(time="time",value="b_NRN",data=primary_dataframe,unit="trial",condition='drug',ax=ax5,ci=95)
    sns.tsplot(time="time",value="c_NRN",data=primary_dataframe,unit="trial",condition='drug',ax=ax6,ci=95)
    ax1.set(xlabel='',ylabel='A, neurons=%s'%P['n_neurons_A'])
    ax2.set(xlabel='',ylabel='B_LIF, neurons=%s'%P['n_neurons_B_LIF'])
    ax3.set(xlabel='',ylabel='C_LIF, neurons=%s'%P['n_neurons_C_LIF'])
    ax4.set(xlabel='',ylabel='A, neurons=%s'%P['n_neurons_A'])
    ax5.set(xlabel='',ylabel='B_NRN, neurons=%s'%P['n_neurons_B_NRN'])
    ax6.set(xlabel='',ylabel='C_NRN, neurons=%s'%P['n_neurons_C_NRN'])
    plt.tight_layout()

    figure.savefig('plot.png')
    primary_dataframe.to_pickle('data.pkl')
    param_df=pd.DataFrame([P])
    param_df.reset_index().to_json('parameters.json',orient='records')

def main():
    import pandas as pd
    import numpy as np
    import copy
    from multiprocessing import Pool
    from pathos.multiprocessing import ProcessingPool as Pool2

    P=eval(open('params_ndn_test.txt').read())
    exp_params=[]
    for drug in P['drugs']:
        for trial in np.arange(0,P['trials']):
            my_P = copy.copy(P)
            my_P['drug']=drug
            my_P['trial']=trial
            exp_params.append(my_P)
    # df=run(exp_params[0])

    init_directory()
    pool = Pool2(nodes=P['threads'])

    df_list = pool.map(run, exp_params)
    primary_dataframe = pd.concat([df for df in df_list], ignore_index=True)
    export(P,primary_dataframe)

if __name__=='__main__':
    main()