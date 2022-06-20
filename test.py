from simulate.simulate_dynamic2 import simulate_dynamic2
from modellers.analyze_consensusHMM import analyze_consensusHMM
import matplotlib.pyplot as plt
import numpy as np

traces,vits,chains= simulate_dynamic2(1,1,200,1000)
'''
print (traces[20][0:10])
plt.plot(traces[20])
plt.show()

plt.plot(chains[20], 'k')
plt.show()

plt.hist(np.concatenate(traces),bins = 100)
plt.show()
'''

res = analyze_consensusHMM(traces, 2)

print(res.mean)
print(res.tmatrix/(res.tmatrix.sum(1)[:,None]))
