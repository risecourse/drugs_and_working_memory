import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

fname='wm_mp_choice_inject_5WBGDPJBN'
empirical_dataframe=pd.read_pickle('empirical_data_start2s') #add '.pkl' onto filename for windows
# primary_dataframe=pd.read_pickle('data/'+fname+'_primary_data.pkl')

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

sns.set(context='poster')
figure, (ax1,ax2) = plt.subplots(2, 1)
sns.tsplot(time="time",value="accuracy",data= empirical_dataframe.query("drug=='control (empirical)'"),
			unit='trial',condition='drug',
			interpolate=False,ax=ax2,color=sns.color_palette('dark'))
sns.tsplot(time="time",value="accuracy",data= empirical_dataframe.query("drug=='control (empirical)'"),
			unit='trial',condition='drug',
			interpolate=True,ax=ax2,color=sns.color_palette('dark'),legend=False)
ax2.set(xlabel='delay period length (s)',xlim=(0,9.5),ylim=(0.5,1),ylabel='DRT accuracy')
plt.legend(loc='lower left')
figure.savefig('empirical_plot_all.png')
plt.show()

# empirical_dataframe.query("drug=='control (empirical)'")