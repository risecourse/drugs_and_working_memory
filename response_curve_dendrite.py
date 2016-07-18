"""Script for generating the F-I-curve for stimulation in the soma and store
the data as plot and npz file for use in the nrnengo module."""

import neuron
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

neuron.h.load_file('/home/pduggins/nengo_detailed_neurons/models/bahr2.hoc')
neuron.h.dt = 0.005
current = np.linspace(0.0, 3.0, 100)
n_locations=50
locations = np.linspace(0,1.0,n_locations)

def get_current(i,l):
	#current increases from i at soma to 2i at distal end of apical dendrite
	return i*(locations[l]/locations[-1]+1)/n_locations

def reset_channels(drug):
    #strongly enhance the I_h current, by opening HCN channels, to create shunting under control
    cell.tuft.gbar_ih *= drug_effect_channel[drug]
    cell.apical.gbar_ih *= drug_effect_channel[drug]
    cell.recalculate_channel_densities()

drug_effect_channel={'control':1.0,'PHE':1.05,'GFC':0.85}
drug='GFC'

freq = []
iclamp = []
cell = neuron.h.Bahr2()
for l in range(len(locations)):
	iclamp.append(neuron.h.IClamp(cell.apical(locations[l])))
	iclamp[l].delay = 200
	iclamp[l].dur = 1100
v = neuron.h.Vector()
v.record(cell.soma(0.5)._ref_v)
apcount = neuron.h.APCount(cell.soma(0.5))
spikes = neuron.h.Vector()
apcount.record(neuron.h.ref(spikes))

for i in current:
	print 'running drug %s, current %s' %(drug,i)
	for l in range(len(locations)):
		iclamp[l].amp = get_current(i,l)
	neuron.init()
	reset_channels(drug)
	neuron.run(1500)
	spike_array = np.array(spikes)
	freq.append(np.sum(spike_array > 300))

sns.set(context='poster')
figure, ax1 = plt.subplots(1,1)
ax1.plot(current, freq)
ax1.set(xlabel="Injected current (nA, along apical dendrite)", ylabel="Firing rate (1/s)")
figure.savefig('response_curve_dendrite_'+drug+'_'+str(drug_effect_channel[drug])+'.png')

# np.savez('/home/pduggins/nengo_detailed_neurons/data/bahl2_response_curve_dendrite.npz', current=current, rate=freq)

