import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from .generate_trace import generate_chain, add_emission

def simulate_dynamic2(rep_number,nrestarts,nmols,nt):
	mu =  np.array([0.05, 0.95, 0.05, 0.95])
	s =  np.array([0.1, 0.1, 0.1, 0.1])
	pi = np.array([0.2, 0.4/3, 0.6, 0.4])



	transition = np.array([[0.94, 0.06, 0.002, 0.002],
						   [0.09, 0.91, 0.002, 0.002],
						   [0.002/3, 0.002/3, 0.98, 0.02],
						   [0.002/3, 0.002/3, 0.03, 0.97]])


	traces = []
	vits = []
	chains = []

	N = nmols
	T = nt
	K = np.arange(len(mu), dtype = 'float64')

	dataset_number = 3

	seed = nrestarts*dataset_number+rep_number
	for j in range(N):

		c =  generate_chain(T,K,pi, transition, 666 + seed + j)
		i,t = add_emission(c,K,mu,s)

		chains.append(c)
		traces.append(t)
		vits.append(i)

	chains = np.array(chains)
	traces = np.array(traces)
	vits = np.array(vits)

	return traces, vits, chains

if __name__ == '__main__':

	traces,vits,chains = simulate_dynamic2(1,1,200,1000)
	print (traces[10][0:10])
	plt.plot(traces[10])
	plt.show()

	plt.plot(chains[10], 'k')
	plt.show()


	plt.hist(np.concatenate(traces),bins = 100)
	plt.show()
