import pytry
import nengo
import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

import trial

drugs = ['control','PHE','GFC'] # ['GFC'] # 
plot_firing_rate = False
n_trials = 50
n_neurons_wm = 100
n_s = 0

# Plotting agrees with bioneuron_oracle plot styles
palette = sns.hls_palette(9, h=.6, l=.3, s=.9)
# palette = sns.palplot(sns.color_palette("bright", 9))
colors = ["blue", "red", "green", "yellow", "purple", "orange", "cyan", "hot pink", "tan"]
palette = sns.xkcd_palette(colors)
sns.set(context='poster', palette=palette, style='whitegrid')

drug_stim = {'control':0.0, 'PHE':-0.2, 'GFC':0.5}
# drug_gainbias = {'control':[1.0,1,0], 'PHE':[0.99,1.02], 'GFC':[1.05,0.95]}
drug_gainbias = {'control':[1.0,1,0], 'PHE':[0.98,1.03], 'GFC':[1.05,0.95]}
# drug_functional = {'control':1.0, 'PHE':0.985, 'GFC':1.03}
drug_functional = {'control':1.0, 'PHE':0.99, 'GFC':1.02}
drug_biophysical = {'control':500, 'PHE':100, 'GFC':1000}

""" Empirical dataframe """
q=0
emp_times = [3.0,5.0,7.0,9.0]
df_emp = pandas.DataFrame(
	columns=('time','drug','correct','trial'),
	index=np.arange(0, 12))
pre_PHE=[0.972, 0.947, 0.913, 0.798]
pre_GFC=[0.970, 0.942, 0.882, 0.766]
post_GFC=[0.966, 0.928, 0.906, 0.838]
post_PHE=[0.972, 0.938, 0.847, 0.666]
for t in range(len(emp_times)):
	df_emp.loc[q]=[emp_times[t],'control (empirical)',np.average([pre_GFC[t],pre_PHE[t]]),0]
	df_emp.loc[q+1]=[emp_times[t],'PHE (empirical)',post_PHE[t],0]
	df_emp.loc[q+2]=[emp_times[t],'GFC (empirical)',post_GFC[t],0]
	q+=3

seeds = np.arange(1 + n_s*n_trials, 1 + n_s*n_trials + n_trials)
prob_misperceive = 0.05
tau_smooth = 0.01
enc_min = 0.3
enc_max = 0.6
rng = np.random.RandomState(seed=seeds[0])
cues = rng.choice([-1, 1], p=[0.5, 0.5], size=seeds.shape)
perceived = rng.choice([False, True],
	p=[prob_misperceive, 1.0 - prob_misperceive], size=seeds.shape)


""" Drug Experiment """
plot_dir = 'pytry_data/biophysical_july26/'
# for drug in drugs:
# 	data_dir = plot_dir + drug
# 	for a, seed in enumerate(seeds):
# 		print 'running drug %s, trial=%s...' %(drug, seed)
# 		trial.Trial().run(
# 			seed=seed,
# 			cue=cues[a],
# 			perceived=perceived[a],
# 			prob_misperceive=prob_misperceive,
# 			n_neurons_wm=n_neurons_wm,
# 			data_dir=data_dir,
# 			plot_dir=plot_dir,
# 			# drug_functional=drug_functional[drug],
# 			# drug_stim=drug_stim[drug],
# 			# drug_gainbias=drug_gainbias[drug],
# 			drug_biophysical=drug_biophysical[drug],
# 			noise_decision=0.025,
# 			k_time=0.4,
# 			radius=1.0,
# 			verbose=False,
# 			dt_sample=0.001,
# 			t_train=4.0,
# 			t_test=4.0,
# 			t_delay=8.0,
# 			JL_dims=3,
# 			JL_dims_mag=2e-4,
# 			neuron_type='Bioneuron', # Bioneuron, LIF
# 			data_format='npz',
# 			decision_type='basal_ganglia',  # 'default',
# 			signal_train_oracle='ramp',
# 			signal_readout_oracle='ramp',
# 			signal_test_oracle='ramp'
# 			)

""" Dataframe construction """
import time as time_py
# start_time = time_py.time()
times = pytry.read(plot_dir+'control')[0]['time']
columns = ('seed', 'drug', 'time', 'wm', 'correct')
df = pandas.DataFrame(columns=columns)
# df_list = []
drugs = ['control','PHE','GFC']
for d, drug in enumerate(drugs):
	data_dir = plot_dir + drug
	data = pytry.read(data_dir)
	for s, seed in enumerate(seeds):
		df_time_list = []
		print 'adding drug %s, trial=%s to DRT dataframe...' %(drug, seed)
		for t, time in enumerate(times):
			time = data[s]['time'][t]
			cue = data[s]['cue']
			wm = data[s]['wm'][t][0]
			output = data[s]['output'][t][0]
			wm_scaled = wm * cue  # force wm from different cues to have the same sign
			correct = 1.0 * (np.sign(output) == np.sign(cue))
			df_temp = pandas.DataFrame(
				[[seed, drug + ' (model)', time, wm_scaled, correct]], columns=columns)
			df_time_list.append(df_temp)
			del df_temp
		df_seed = pandas.concat(df_time_list, ignore_index=True)
		df = pandas.concat([df, df_seed], ignore_index=True)
		del df_time_list

print 'Plotting DRT...'
""" DRT accuracy plot """
fig1, (ax1, ax2) = plt.subplots(2,1, sharex=True)
sns.tsplot(time='time', value='wm', unit='seed', condition='drug',
	data=df, ax=ax1, ci=95, legend=False)
ax1.set(ylabel='$|\hat{x}_0(t)|$')
sns.tsplot(time='time', value='correct', unit='seed', condition='drug',
	data=df, ax=ax2, ci=95, alpha=0.5)
sns.tsplot(time='time', value='correct', unit='trial', condition='drug',
	data=df_emp, ax=ax2, ci=95, marker='o')
ax2.set(xlabel='time (s)', ylabel='DRT accuracy', xlim=((0,10)), ylim=((0.5,1.0)))
ax2.legend(loc='lower left', title='drug')
fig1.savefig(plot_dir + 'drt.png')

import sys
if not plot_firing_rate: sys.exit()

""" Firing rate dataframe """
j=0
lpf = nengo.Lowpass(tau_smooth)
columns=('seed', 'drug', 'time', 'neuron-trial', 'encoder', 'tuning', 'rate')
# df_firing = pandas.DataFrame(
# 	columns=('seed', 'drug', 'time', 'neuron-trial', 'encoder', 'tuning', 'rate'),
# 	index=np.arange(0, seeds.shape[0] * times.shape[0] * n_neurons_wm))
# df_list = []
df_firing = pandas.DataFrame(columns=columns)
times = pytry.read(plot_dir+'control')[0]['time']
for d, drug in enumerate(drugs):
	data_dir = plot_dir + drug
	data = pytry.read(data_dir)
	for s, seed in enumerate(seeds):
		df_time_list = []
		print 'adding drug %s, trial=%s to firing rate dataframe...' %(drug, seed)
		rates = lpf.filt(data[s]['spikes'])
		for t, time in enumerate(times):
			for n in range(n_neurons_wm):
				time = data[s]['time'][t]
				rate = np.around(rates[t][n], decimals=5)  # otherwise goes to e-168
				enc = data[s]['encoder'][n][0]  # cue dimension
				cue = data[s]['cue']
				if np.sign(cue) == np.sign(enc):
					if abs(enc) < enc_min:
						tuning = 'weak'
					elif enc_min < abs(enc) < enc_max:
						tuning = 'moderate'
					elif abs(enc) > enc_max:
						tuning = 'strong'
				else:
					tuning = 'nonpreferred'
				df_temp = pandas.DataFrame(
					[[seed, drug, time, n+s*n_neurons_wm, enc, tuning, rate]], columns=columns)
				df_time_list.append(df_temp)
				del df_temp
				# df_firing.loc[j] = [seed, drug, time, n+s*n_neurons_wm, enc, tuning, rate]
				# j+=1
		df_seed = pandas.concat(df_time_list, ignore_index=True)
		df_firing = pandas.concat([df_firing, df_seed], ignore_index=True)
		del df_time_list

df_moderate = pandas.DataFrame(df_firing.query("tuning=='moderate'")).reset_index()
df_nonpreferred = pandas.DataFrame(df_firing.query("tuning=='nonpreferred'")).reset_index()

""" Firing rate plot """
print 'Plotting Firing Rate...'
fig2, (ax3, ax4) = plt.subplots(1,2, sharey=True)
sns.tsplot(time='time', value='rate', unit='neuron-trial', condition='drug',
	data=df_moderate, ax=ax3, ci=95)  # .reset_index()
sns.tsplot(time='time', value='rate', unit='neuron-trial', condition='drug',
	data=df_nonpreferred, ax=ax4, ci=95, legend=False)
ax3.set(xlabel='time (s)', ylabel='Firing Rate (Hz)', title='preferred')
ax4.set(xlabel='time (s)', title='nonpreferred')
fig2.savefig(plot_dir + 'firing.png')