import numpy as np
from .generate_trace import generate_chain, add_emission

def simulate_dynamic2(rep_number,nrestarts,nmols,nt,snr,truncate=None):
	dataset_number = 3

	mu =  np.array([0.0, 1., 0., 1.])
	s =  np.array([1., 1., 1., 1.])/snr
	pi = np.array([0.2, 0.4/3, 0.6, 0.4])

	transition = np.array([
		[0.94, 0.06, 0.002, 0.002],
		[0.09, 0.91, 0.002, 0.002],
		[0.002/3, 0.002/3, 0.98, 0.02],
		[0.002/3, 0.002/3, 0.03, 0.97]])


	traces = []
	vits = []
	chains = []

	N = nmols
	T = nt
	K = np.arange(len(mu), dtype = 'float64')

	seed = (nrestarts*nmols)*dataset_number+rep_number*nmols
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
	traces,vits,chains = simulate_dynamic2(0,1,200,1000,5.,500.)
	print (traces[100][0:10])
	plt.plot(traces[100])
	#plt.plot(vits[20], 'k')
	plt.plot(chains[100], 'k')
	plt.show()


	plt.hist(np.concatenate(traces),bins = 100)
	plt.show()
