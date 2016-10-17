import nengo
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from helper import empirical_dataframe


'''Plot Model DRT Data'''
root=os.getcwd()
datadir='3CTIRK57V' #'YOUR DIRECTORY HERE'
if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
	change_to=root+'/data/'+datadir+'/' #linux or mac
elif sys.platform == "win32":
	change_to=root+'\\data\\'+datadir+'\\' #windows
os.chdir(change_to)
primary_dataframe=pd.read_pickle('primary_data.pkl')
firing_dataframe=pd.read_pickle('firing_data.pkl')
emp_dataframe=empirical_dataframe()
sns.set(context='poster')

figure, (ax1, ax2) = plt.subplots(2, 1)
sns.tsplot(time="time",value="wm",data=primary_dataframe,unit="trial",condition='drug',ax=ax1,ci=95)
sns.tsplot(time="time",value="correct",data=primary_dataframe,unit="trial",condition='drug',ax=ax2,ci=95)
sns.tsplot(time="time",value="accuracy",data=emp_dataframe,
			unit='trial',condition='drug',
			interpolate=False,ax=ax2,color=sns.color_palette('dark'))
sns.tsplot(time="time",value="accuracy",data=emp_dataframe,
			unit='trial',condition='drug',
			interpolate=True,ax=ax2,color=sns.color_palette('dark'),legend=False)
ax1.set(xlabel='',ylabel='decoded $\hat{cue}$',xlim=(0,9.5),ylim=(0,1))
			# ,title="drug_type=%s, decision_type=%s, trials=%s" %(drug_type,decision_type,n_trials))
ax2.set(xlabel='time (s)',xlim=(0,9.5),ylim=(0.5,1),ylabel='DRT accuracy')
figure.savefig('primary_plots_replotted.png')

figure2, (ax3, ax4) = plt.subplots(1, 2)
if len(firing_dataframe.query("tuning=='weak'"))>0:
	sns.tsplot(time="time",value="firing_rate",unit="neuron-trial",condition='drug',ax=ax3,ci=95,
			data=firing_dataframe.query("tuning=='weak'").reset_index())
if len(firing_dataframe.query("tuning=='nonpreferred'"))>0:
	sns.tsplot(time="time",value="firing_rate",unit="neuron-trial",condition='drug',ax=ax4,ci=95,
			data=firing_dataframe.query("tuning=='nonpreferred'").reset_index())
ax3.set(xlabel='time (s)',xlim=(0.0,9.5),ylim=(0,250),ylabel='Normalized Firing Rate',title='Preferred Direction')
ax4.set(xlabel='time (s)',xlim=(0.0,9.5),ylim=(0,250),ylabel='',title='Nonpreferred Direction')
figure2.savefig('firing_plots_replotted.png')



'''Plot Empirical DRT Data'''
emp_dataframe=empirical_dataframe()
sns.set(context='poster')
figure, (ax2) = plt.subplots(1, 1)
sns.tsplot(time="time",value="accuracy",data= emp_dataframe,
			unit='trial',condition='drug',
			interpolate=False,ax=ax2,color=sns.color_palette('dark'))
sns.tsplot(time="time",value="accuracy",data= emp_dataframe,
			unit='trial',condition='drug',
			interpolate=True,ax=ax2,color=sns.color_palette('dark'),legend=False)
ax2.set(xlabel='delay period length (s)',xlim=(0,9.5),ylim=(0.5,1),ylabel='DRT accuracy')
plt.legend(loc='lower left')




'''LIF Tuning Curves with modified gain and bias'''
sns.set(context='poster')
figure, (ax1,ax2) = plt.subplots(2, 1)
n = nengo.neurons.LIFRate(tau_rc=0.02, tau_ref=0.002) #n is a Nengo LIF neuron, these are defaults
J = np.linspace(1,3,100)
ax1.plot(J, n.rates(J, gain=1, bias=-1),label="control: gain=1, bias=-1")
ax1.plot(J, n.rates(J, gain=0.99, bias=-1.02),label="PHE: gain=0.99, bias=-1.02") 
ax1.plot(J, n.rates(J, gain=1.05, bias=-0.95),label="GFC: gain=1.05, bias=-0.95") 
ax1.set(xlabel='input current',ylabel='activity (Hz)')
ax1.legend(loc='upper left')
figure.savefig('alpha_bias_tuning_curves.png')
