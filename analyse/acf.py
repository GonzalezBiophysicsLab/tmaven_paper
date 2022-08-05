import numpy as np
import numba as nb


@nb.njit
def gen_mc_acf(tau,nsteps,tmatrix,mu,var,ppi):
	## Using a transition probability matrix, not a rate matrix, or Q matrix
	## because this comes straight out of an HMM.... so not quick exact

	nstates,_ = tmatrix.shape
	pi0 = np.eye(nstates)

	## get steady state probabilities
	pinf = np.linalg.matrix_power(tmatrix.T,100*int(1./tmatrix.min()))[:,0]

	## use fluctuations
	mubar =  (pinf*mu).sum()
	mm = mu - mubar

	### expectation here
	E_y0yt = np.zeros(nsteps)
	for i in range(nstates): # loop over initial state
		for j in range(nstates): # loop over final state
			for k in range(nsteps): # loop over time delay steps
				## E[y_0*t_t] = \sum_ij m_i * m_j * (A^n \cdot \delta (P_i))_j * P_inf,i
				E_y0yt[k] += mm[i]*mm[j] * (np.dot(np.linalg.matrix_power(tmatrix.T,k),pi0[i])[j]) * pinf[i]

	## add gaussian noise terms
	for i in range(nstates):
		E_y0yt[0] += var[i]*pinf[i]
	## normalize
	E_y0yt /= E_y0yt[0]

	t = tau*np.arange(nsteps)
	return t,E_y0yt
