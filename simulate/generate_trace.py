import numpy as np
import matplotlib.pyplot as plt
import numba as nb

@nb.njit
def numba_choice(K, pi):
	x = np.random.rand()

	y = np.cumsum(pi)/pi.sum()

	for i in range(len(y)):
		if x < y[i]:
			return K[i]

@nb.njit(nb.double[:](nb.int64,nb.double[:],nb.double[:],nb.double[:,:], nb.int64))
def generate_chain(N,K,pi,tm,seed):
    np.random.seed(seed)
    chain = np.ones(N)
    state = numba_choice(K, pi)
    for i in range(N):
        chain[i] = state
        trans = tm[int(state)]
        state = numba_choice(K, trans)

    return chain


@nb.njit
def add_emission(chain, K, mu, s):
    traj = np.ones_like(chain)
    id = traj.copy()
    for i in range(len(chain)):
        id[i] = mu[int(chain[i])]
        traj[i] = id[i] + np.random.normal(loc = 0, scale = s[int(chain[i])])
    return id,traj

def generate_dwells(trace, dwell_list, means):
    trace = trace[~np.isnan(trace)]
    #print(trace)
    if len(trace) > 0: #proetcting if all is NaN
        dwell_split = np.split(trace, np.argwhere(np.diff(trace)!=0).flatten()+1)

        if len(dwell_split) > 2: #protecting against no or single transition in a trace
            dwell_split = dwell_split[1:-1] #skipping first and last dwells
            for d in dwell_split:
                ind = int(np.argwhere(d[0] == means))
                dwell_list[str(ind)].append(len(d))

    return dwell_list

if __name__ == '__main__':
    mu =  np.array([0.1, 0.5, 0.9])
    s =  np.array([0.01, 0.05, 0.09])
    pi = np.array([0.1, 0.3, 0.6])
    transition = np.array([[0.9, 0.05, 0.05],
                           [0.08, 0.9, 0.02],
                           [0.04,0.06,0.9]])

    c =  generate_chain(200,np.arange(3, dtype='float'),pi, transition)
    i,t = add_emission(c,3,mu,s)

    print (t[0:10])
    plt.plot(t)
    plt.plot(c, 'k')
    plt.show()

    d = {str(i):[] for i in range(len(mu))}
    d = generate_dwells(i,d,mu)
    print(d)
