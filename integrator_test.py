import nengo
import numpy as np
import matplotlib.pyplot as plt

model=nengo.Network()
with model:
	stim=nengo.Node(output=lambda t: 1.0*(t<0.5))
	ramp=nengo.Node(output=lambda t: 0.0*(t>0.5))
	A=nengo.Ensemble(200,dimensions=2)
	B=nengo.Ensemble(200,dimensions=2)
	nengo.Connection(stim,A[0])
	nengo.Connection(ramp,A[1])
	nengo.Connection(A,B,synapse=0.1,transform=0.1)
	nengo.Connection(B,B,synapse=0.1)
	probe=nengo.Probe(B[0],synapse=0.01,sample_every=0.005)

sim=nengo.Simulator(model)
sim.run(1.0)
figure, ax1 = plt.subplots(1,1)
x=np.arange(0,1.0,0.005)
y=sim.data[probe]
ax1.plot(x,y)
plt.show()