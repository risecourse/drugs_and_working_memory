#make and export pd dataframe for empirical data
import pandas as pd
import numpy as np
columns=('time','drug','accuracy','trial')
emp_timesteps = [3.0,5.0,7.0,9.0]
empirical_dataframe = pd.DataFrame(columns=columns,index=np.arange(0, 12))
pre_PHE=[0.972, 0.947, 0.913, 0.798]
pre_GFC=[0.970, 0.942, 0.882, 0.766]
post_GFC=[0.966, 0.928, 0.906, 0.838]
post_PHE=[0.972, 0.938, 0.847, 0.666]
q=0
for t in range(len(emp_timesteps)):
	empirical_dataframe.loc[q]=[emp_timesteps[t],'control (empirical)',np.average([pre_GFC[t],pre_PHE[t]]),0]
	empirical_dataframe.loc[q+1]=[emp_timesteps[t],'PHE (empirical)',post_PHE[t],0]
	empirical_dataframe.loc[q+2]=[emp_timesteps[t],'GFC (empirical)',post_GFC[t],0]
	q+=3
empirical_dataframe.to_pickle('empirical_data')