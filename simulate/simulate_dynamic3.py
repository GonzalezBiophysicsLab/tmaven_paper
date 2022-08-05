import numpy as np
from .generate_trace import generate_chain, add_emission

def simulate_dynamic3(rep_number,nrestarts,nmols,nt,snr,truncate=None):
	dataset_number = 4

	mu =  np.array([0.0, 0.75, 1., 0.0, 1.])
	s =  np.array([1., 1., 1., 1., 1.])/snr
	pi = np.array([0.2, 0.5/3, 0.4/3, 0.6, 0.4])



	transition = np.array([
		[0.980, 0.010, 0.010, 0.008/6, 0.008/6],
		[0.0125, 0.975, 0.0125, 0.008/6, 0.008/6],
		[0.015, 0.015, 0.970, 0.008/6, 0.008/6],
		[0.008/18, 0.008/18, 0.008/18, 0.98, 0.02],
		[0.008/18, 0.008/18, 0.008/18, 0.03, 0.97]])

	chains = []
	traces = []
	vits = []

	N = nmols
	T = nt
	K = np.arange(len(mu), dtype = 'float64')

	seed = 666 + nrestarts*dataset_number+rep_number
	for j in range(N):

		c =  generate_chain(T,K,pi, transition, seed + j)
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
	import matplotlib.pyplot as plt
	traces,vits,chains = simulate_dynamic3(1,1,200,1000,5.,500.)
	print (traces[100][0:10])
	plt.plot(traces[100])
	#plt.plot(vits[20], 'k')
	plt.plot(chains[100], 'k')
	plt.show()


	plt.hist(np.concatenate(traces),bins = 100)
	plt.show()
