import numpy as np
from .generate_trace import generate_chain, add_emission

def simulate_static3(rep_number,nrestarts,nmols,nt,snr,truncate=None,prop=0.25):
	dataset_number = 2

	mu1 =  np.array([0.0, 0.75, 1.])
	s1 =  np.array([1., 1., 1.])/snr
	pi1 = np.array([0.6, 0.5, 0.4])

	transition1 = np.array([[0.980, 0.010, 0.010],
							[0.0125, 0.975, 0.0125],
							[0.015, 0.015, 0.970]])

	mu2 =  np.array([0.0, 1.])
	s2 =  np.array([1., 1.])/snr
	pi2 = np.array([0.6, 0.4])

	transition2 = np.array([[0.98, 0.02],
							[0.03, 0.97]])

	chains = []
	traces = []
	vits = []

	N = nmols
	N1 = int(nmols*prop)
	T = nt
	K1 = np.arange(len(mu1), dtype = 'float64')
	K2 = np.arange(len(mu2), dtype = 'float64')

	seed = (nrestarts*nmols)*dataset_number+rep_number*nmols
	for j in range(N):

		if j < N1:
			c =  generate_chain(T,K1,pi1, transition1, seed + j)
			i,t = add_emission(c,K1,mu1,s1)
		else:
			c =  generate_chain(T,K2,pi2, transition2, seed + j)
			i,t = add_emission(c,K2,mu2,s2)

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
	traces,vits,chains = simulate_static3(1,1,200,1000,5.,500.)
	print (traces[20][0:10])
	plt.plot(traces[20])
	plt.plot(vits[20], 'k')
	plt.show()

	traces = np.array(traces)

	plt.hist(np.concatenate(traces),bins = 100)
	plt.show()
