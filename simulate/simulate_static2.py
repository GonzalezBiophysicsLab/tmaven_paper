import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from generate_trace import generate_chain, add_emission

def simulate_static2(rep_number,nrestarts,nmols,nt,snr,truncate=None):
	dataset_number = 1

	mu =  np.array([0.0, 1.])
	s =  np.array([1., 1.])/snr
	pi = np.array([0.6, 0.3])

	transition1 = np.array([
		[0.96, 0.04],
		[0.08, 0.92]])

	transition2 = np.array([
		[0.98, 0.02],
		[0.04, 0.96]])

	chains = []
	traces = []
	vits = []

	N = nmols
	N1 = nmols//4
	T = nt
	K = np.arange(len(mu), dtype = 'float64')

	seed = 666 + nrestarts*dataset_number+rep_number
	for j in range(N):

		if j < N1:
			c =  generate_chain(T,K,pi, transition1, seed + j)
		else:
			c =  generate_chain(T,K,pi, transition2, seed + j)
		i,t = add_emission(c,K,mu,s)

		if not truncate is None:
			np.random.seed(seed+j)
			pbt = int(np.random.exponential(truncate))
			if pbt < 1:
				pbt = 1
			if pbt >= c.size:
				pbt = -1
			c[pbt:] = np.nan
			i[pbt:] = np.nan
			t[pbt:] = np.nan

		chains.append(c)
		traces.append(t)
		vits.append(i)

	chains = np.array(chains)
	traces = np.array(traces)
	vits = np.array(vits)

	return traces, vits, chains

if __name__ == '__main__':

	traces,vits,chains = simulate_static2(1,1,200,1000,5.,500.)
	print (traces[-20][0:10])
	plt.plot(traces[-20])
	plt.plot(vits[-20], 'k')
	plt.show()

	traces = np.array(traces)

	plt.hist(np.concatenate(traces),bins = 100)
	plt.show()
