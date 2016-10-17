import nengo
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# fname='wm_mp_choice_recurrent_PF4MXX7WQ'
empirical_dataframe=pd.read_pickle('empirical_data_start2s') #add '.pkl' onto filename for windows
# primary_dataframe=pd.read_pickle('data/'+fname+'_primary_data.pkl')

#representation and forgetting
# sns.set(context='poster')
# figure, (ax1, ax2) = plt.subplots(2, 1)
# sns.tsplot(time="time",value="wm",data=primary_dataframe,unit="trial",condition='drug',ax=ax1,ci=95)
# sns.tsplot(time="time",value="correct",data=primary_dataframe,unit="trial",condition='drug',ax=ax2,ci=95)
# sns.tsplot(time="time",value="accuracy",data=empirical_dataframe,
# 			unit='trial',condition='drug',
# 			interpolate=False,ax=ax2,color=sns.color_palette('dark'))
# sns.tsplot(time="time",value="accuracy",data=empirical_dataframe,
# 			unit='trial',condition='drug',
# 			interpolate=True,ax=ax2,color=sns.color_palette('dark'),legend=False)
# ax1.set(xlabel='',ylabel='decoded $\hat{cue}$',xlim=(0,9.5),ylim=(0,1))
# 			# ,title="drug_type=%s, decision_type=%s, trials=%s" %(drug_type,decision_type,n_trials))
# ax2.set(xlabel='time (s)',xlim=(0,9.5),ylim=(0.5,1),ylabel='DRT accuracy')
# figure.savefig(fname+'_primary_plots_2.png')
# plt.show()

#empirical data
sns.set(context='poster')
figure, (ax2) = plt.subplots(1, 1)
sns.tsplot(time="time",value="accuracy",data= empirical_dataframe,
			unit='trial',condition='drug',
			interpolate=False,ax=ax2,color=sns.color_palette('dark'))
sns.tsplot(time="time",value="accuracy",data= empirical_dataframe,
			unit='trial',condition='drug',
			interpolate=True,ax=ax2,color=sns.color_palette('dark'),legend=False)
ax2.set(xlabel='delay period length (s)',xlim=(0,9.5),ylim=(0.5,1),ylabel='DRT accuracy')
plt.legend(loc='lower left')
figure.savefig('empirical_plot_all.png')
plt.show()

# empirical_dataframe.query("drug=='control (empirical)'")

#Tuning Curves with alpha/bias
#'drug_effect_gainbias':{'control':[1.0,1,0],'PHE':[0.99,1.02],'GFC':[1.05,0.95]}
# sns.set(context='poster')
# figure, (ax1,ax2) = plt.subplots(2, 1)
# n = nengo.neurons.LIFRate(tau_rc=0.02, tau_ref=0.002) #n is a Nengo LIF neuron, these are defaults
# J = np.linspace(1,3,100)
# ax1.plot(J, n.rates(J, gain=1, bias=-1),label="control: gain=1, bias=-1")
# ax1.plot(J, n.rates(J, gain=0.99, bias=-1.02),label="PHE: gain=0.99, bias=-1.02") 
# ax1.plot(J, n.rates(J, gain=1.05, bias=-0.95),label="GFC: gain=1.05, bias=-0.95") 
# ax1.set(xlabel='input current',ylabel='activity (Hz)')
# ax1.legend(loc='upper left')
# figure.savefig('alpha_bias_tuning_curves.png')
# plt.show()