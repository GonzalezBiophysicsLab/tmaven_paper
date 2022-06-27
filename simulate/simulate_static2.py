import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from .generate_trace import generate_chain, add_emission

def simulate_static2(rep_number,nrestarts,nmols,nt):
	mu =  np.array([0.05, 0.95])
	s =  np.array([0.1, 0.1])
	pi = np.array([0.6, 0.4])

	transition1 = np.array([[0.94, 0.06],
							[0.09, 0.91]])

	transition2 = np.array([[0.98, 0.02],
							[0.03, 0.97]])

	traces = []
	vits = []
	chains = []

	N1 = nmols//4
	T = nt
	K = np.arange(len(mu), dtype = 'float64')

	dataset_number = 1

	seed = nrestarts*dataset_number+rep_number
	for j in range(N1):

		c =  generate_chain(T,K,pi, transition1, 666 + seed + j)
		i,t = add_emission(c,K,mu,s)

		chains.append(c)
		traces.append(t)
		vits.append(i)

	for jj in range(nmols - N1):

		c =  generate_chain(T,K,pi, transition2, 666 + seed + N1 + jj)
		i,t = add_emission(c,K,mu,s)

		chains.append(c)
		traces.append(t)
		vits.append(i)

	chains = np.array(chains)
	traces = np.array(traces)
	vits = np.array(vits)

	return traces, vits, chains

if __name__ == '__main__':

	traces,vits,chains= simulate_static2(1,1,200,1000)
	print (traces[40][0:10])
	plt.plot(traces[40])
	plt.show()

	plt.plot(vits[40], 'k')
	plt.show()

	print (traces[-40][0:10])
	plt.plot(traces[-40])
	plt.show()

	plt.plot(vits[-40], 'k')
	plt.show()

	plt.hist(np.concatenate(traces),bins = 100)
	plt.show()
