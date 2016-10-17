'''
Figure 2: Compare an NEF optimal integrator with one
that has the added instabilities described in the text
'''


import nengo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nengo.utils.matplotlib import rasterplot

noise_wm=1e-9
ramp_mag=0.0
model=nengo.Network()
with model:
	stim=nengo.Node(output=lambda t: 1.0*(t<1.0))
	ramp=nengo.Node(output=lambda t: ramp_mag*(t>1.0))
	noise_wm_node = nengo.Node(output=np.random.normal(0.0,noise_wm))
	A=nengo.Ensemble(100,dimensions=2)
	B=nengo.Ensemble(1000,dimensions=2) #,max_rates=nengo.dists.Uniform(100,200)
	nengo.Connection(stim,A[0])
	nengo.Connection(ramp,A[1])
	nengo.Connection(A,B,synapse=0.1,transform=0.1)
	nengo.Connection(B,B,synapse=0.1)
	probe=nengo.Probe(B[0],synapse=0.01)
	p_spikes=nengo.Probe(B.neurons,'spikes')

sim=nengo.Simulator(model)
sim.run(8.0)
sns.set(context='poster')
figure=plt.figure()
t=sim.trange()
y=sim.data[probe]
#plot 100 neurons with most spikes
spikes=sim.data[p_spikes]
spike_sum=np.sum(spikes,axis=0)
indices=np.argsort(spike_sum)[::-1]
top_spikes=spikes[:,indices[0:100]]
ax1=figure.add_subplot(223)
ax1.plot(t,y)
with sns.axes_style("white"):
	ax3=figure.add_subplot(221)
	rasterplot(t, top_spikes, ax=ax3, use_eventplot=True, color='k', linewidth=0.1)
ax1.set(xlabel='time (s)',ylim=(0.0,1.0),
	ylabel='represented \nvalue $\hat{x}$')
ax3.set(ylabel='neuron \nactivity $a_i(t)$',title='optimal integrator')
ax3=plt.gca()
ax3.invert_yaxis()

noise_wm=0.005
ramp_mag=0.4
sim.reset()
sim.run(8.0)
y=sim.data[probe]
#plot the 100 neurons from above
spikes=sim.data[p_spikes]
top_spikes=spikes[:,indices[0:100]]
ax2=figure.add_subplot(224)
ax2.plot(t,y)
with sns.axes_style("white"):
	ax4=figure.add_subplot(222)
	rasterplot(t, top_spikes, ax=ax4, use_eventplot=True, color='k', linewidth=0.1)
ax2.set(xlabel='time (s)',ylim=(0.0,1.0))
ax4.set(title='added instabilities')
ax4=plt.gca()
ax4.invert_yaxis()
plt.show()