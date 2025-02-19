import numpy as np
from .generate_trace import generate_chain, add_emission

def simulate_hetstat(rep_number,nrestarts,nmols,nt,snr, prop=0.25, **extra):
	dataset_number = 1

	mu =  np.array([0.0, 1.])
	s =  np.array([1., 1.])/snr
	pi = np.array([0.6, 0.4])

	transition1 = np.array([
		[0.94, 0.06],
		[0.09, 0.91]])

	transition2 = np.array([
		[0.98, 0.02],
		[0.03, 0.97]])

	chains = []
	traces = []
	vits = []

	N = nmols
	N1 = int(nmols*prop)
	T = nt
	K = np.arange(len(mu), dtype = 'float64')

	seed = (nrestarts*nmols)*dataset_number+rep_number*nmols
	for j in range(N):

		if j < N1:
			c =  generate_chain(T,K,pi, transition1, seed + j)
		else:
			c =  generate_chain(T,K,pi, transition2, seed + j)
		i,t = add_emission(c,K,mu,s)

		chains.append(c)
		traces.append(t)
		vits.append(i)

	chains = np.array(chains)
	traces = np.array(traces)
	vits = np.array(vits)

	return traces, vits, chains

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	traces,vits,chains = simulate_static2(1,1,200,1000,5.,500.)
	print (traces[-20][0:10])
	plt.plot(traces[-20])
	plt.plot(vits[-20], 'k')
	plt.show()

	traces = np.array(traces)

	plt.hist(np.concatenate(traces),bins = 100)
	plt.show()
