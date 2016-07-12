import neuron
import nrn
import numpy as np
import nengo_detailed_neurons
import matplotlib.pyplot as plt
import seaborn as sns
from nengo_detailed_neurons.neurons import Bahr2, IntFire1

neuron.h.load_file("/home/pduggins/nengo_detailed_neurons/models/bahr2.hoc")
cell = neuron.h.Bahr2()

iclamp = neuron.h.IClamp(cell.tuft(1))
iclamp.delay = 0
iclamp.dur = 2000
iclamp.amp = 0.3

recordings = {}
recordings['time'] = neuron.h.Vector()
recordings['soma(0.5)'] = neuron.h.Vector()
recordings['tuft(1.0)'] = neuron.h.Vector()
recordings['time'].record(neuron.h._ref_t, 0.1)
recordings['soma(0.5)'].record(cell.soma(0.5)._ref_v, 0.1)
recordings['tuft(1.0)'].record(cell.tuft(1.0)._ref_v, 0.1)

#default ih channel properties
times=[400,800,1200,1600]
neuron.init()
neuron.run(times[0])

#strongly enhanced ih channel properties to observe shunting effect
ch_shunting = 100
cell.tuft.gbar_ih *= ch_shunting
cell.apical.gbar_ih *= ch_shunting
cell.recalculate_channel_densities()
neuron.run(times[1])

#slightly enhanced/weakened ih channel properties to observe drug effects
ch_gfc = 0.8
cell.tuft.gbar_ih *= ch_gfc
cell.apical.gbar_ih *= ch_gfc
cell.recalculate_channel_densities()
neuron.run(times[2])
cell.tuft.gbar_ih /= ch_gfc #undo change
cell.apical.gbar_ih /= ch_gfc
cell.recalculate_channel_densities()

ch_phe = 1.15
cell.tuft.gbar_ih *= ch_phe
cell.apical.gbar_ih *= ch_phe
cell.recalculate_channel_densities()
neuron.run(times[3])
cell.tuft.gbar_ih /= ch_phe #undo change
cell.apical.gbar_ih /= ch_phe
cell.recalculate_channel_densities()

x_ticks=np.array(times)-400
x_labels=('normal','enhanced','GFC','PHE')
sns.set(context='poster')
figure, ax = plt.subplots()
ax.plot(recordings['time'],recordings['soma(0.5)'],label='soma')
ax.plot(recordings['time'],recordings['tuft(1.0)'],label='tuft')
ax.set_xlabel('time')
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels)
ax.set_ylabel('voltage')
ax.legend()
figure.savefig('data/Bahr2_channel_spikes.png')

plt.show()