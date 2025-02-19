import numpy as np
import numba as nb

@nb.njit(cache=True)
def numba_choice(K, pi):
	x = np.random.rand()

	y = np.cumsum(pi)/pi.sum()

	for i in range(len(y)):
		if x < y[i]:
			return K[i]

@nb.njit(nb.double[:](nb.int64,nb.double[:],nb.double[:],nb.double[:,:], nb.int64),cache=True)
def generate_chain(N,K,pi,tm,seed):
	np.random.seed(seed)
	chain = np.ones(N)
	state = numba_choice(K, pi)
	for i in range(N):
		chain[i] = state
		trans = tm[int(state)]
		state = numba_choice(K, trans)

	return chain


@nb.njit(cache=True)
def add_emission(chain, K, mu, s):
	traj = np.ones_like(chain)
	id = traj.copy()
	for i in range(len(chain)):
		id[i] = mu[int(chain[i])]
		traj[i] = id[i] + np.random.normal(loc = 0, scale = s[int(chain[i])])
	return id,traj


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	mu =  np.array([0.1, 0.5, 0.9])
	s =  np.array([0.01, 0.05, 0.09])
	pi = np.array([0.1, 0.3, 0.6])
	transition = np.array([
		[0.9, 0.05, 0.05],
		[0.08, 0.9, 0.02],
		[0.04,0.06,0.9]])

	c =  generate_chain(200,np.arange(3, dtype='float'),pi, transition)
	i,t = add_emission(c,3,mu,s)

	print (t[0:10])
	plt.plot(t)
	plt.plot(c, 'k')
	plt.show()

