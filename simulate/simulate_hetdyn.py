import numpy as np
from .generate_trace import generate_chain, add_emission

def simulate_hetdyn(rep_number,nrestarts,nmols,nt,snr,tsh = 4,prop=0.5):
	dataset_number = 3

	mu =  np.array([0.0, 1., 0., 1.])
	s =  np.array([1., 1., 1., 1.])/snr
	Keq = prop/(1-prop)
	pi = np.array([0.6/1., 0.4/1., 0.6/1., 0.4/1.])



	t1 = np.array([            #slow 0-1
		[0.98, 0.02],
		[0.03, 0.97]])

	t2 = np.array([            #fast 0-1
		[0.94, 0.06],
		[0.09, 0.91]])

	t3 = np.array([            #slow-fast
		[1-0.0005*tsh/1., 0.0005*tsh/1.],
		[0.0005*tsh/1., 1-0.0005*tsh/1.]])

	transition = np.array([
		[t1[0][0]*t3[0][0], t1[0][1]*t3[0][0], t2[0][0]*t3[0][1], t2[0][1]*t3[0][1]],
		[t1[1][0]*t3[0][0], t1[1][1]*t3[0][0], t2[1][0]*t3[0][1], t2[1][1]*t3[0][1]],
		[t1[0][0]*t3[1][0], t1[0][1]*t3[1][0], t2[0][0]*t3[1][1], t2[0][1]*t3[1][1]],
		[t1[1][0]*t3[1][0], t1[1][1]*t3[1][0], t2[1][0]*t3[1][1], t2[1][1]*t3[1][1]]]) 


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


		chains.append(c)
		traces.append(t)
		vits.append(i)


	chains = np.array(chains)
	traces = np.array(traces)
	vits = np.array(vits)

	return traces, vits, chains

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	traces,vits,chains = simulate_hetdyn(0,1,200,1000,5.,500.)
	print (traces[100][0:10])
	plt.plot(traces[100])
	#plt.plot(vits[20], 'k')
	plt.plot(chains[100], 'k')
	plt.show()


	plt.hist(np.concatenate(traces),bins = 100)
	plt.show()
