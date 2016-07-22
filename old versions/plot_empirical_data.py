#plot_empirical_data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

empirical_dataframe=pd.read_pickle('empirical_data') #add '.pkl' onto filename for windows
sns.set(context='poster')
figure, ax1 = plt.subplots(1, 1)
sns.tsplot(time="time",value="accuracy",data=empirical_dataframe,unit='trial',condition='drug',
		interpolate=False,ax=ax1)
ax1.set(xlabel='time (s)',xlim=(2,9.5),ylim=(0.5,1),ylabel='accuracy')
figure.savefig('empirical_plot.png')