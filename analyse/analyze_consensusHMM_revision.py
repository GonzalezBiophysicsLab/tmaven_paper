import numpy as np

def run_consensusHMM(dataset, nstates):

	'''
	'modeler.nrestarts':4,
	'modeler.converge':1e-10,
	'modeler.maxiters':1000,
	'modeler.vbconhmm.prior.beta':0.25,
	'modeler.vbconhmm.prior.a':2.5,
	'modeler.vbconhmm.prior.b':0.01,
	'modeler.vbconhmm.prior.alpha':1.,
	'modeler.vbconhmm.prior.pi':1.,
	'''

	new_d = []
	for i in range(len(dataset)):
		yi = dataset[i].astype('double')
		xn = np.where(np.isfinite(yi))[0]
		yi = yi[xn]
		new_d.append(yi)

	mu_prior = np.array([0., 0., 1., 1.,])
	beta_prior = np.ones_like(mu_prior)*0.25
	a_prior = np.ones_like(mu_prior)*2.5
	b_prior = np.ones_like(mu_prior)*0.01
	pi_prior = np.ones_like(mu_prior)*1.
	tm_prior = np.ones((nstates,nstates))*1.

	priors = [mu_prior, beta_prior, a_prior, b_prior, pi_prior, tm_prior]

	maxiters = 1000
	nrestarts = 4
	converge = 1e-10

	mu = np.array([0., 0., 1., 1.,])
	var = np.ones(4)*(1/9)**2
	ppi = np.array([0.3, 0.3, 0.2, 0.2])

	t1 = np.array([            #slow 0-1
		[0.98, 0.02],
		[0.03, 0.97]])

	t2 = np.array([            #fast 0-1
		[0.94, 0.06],
		[0.09, 0.91]])

	tsh = 10 

	t3 = np.array([            #slow-fast
		[1-0.0005*tsh/1., 0.0005*tsh/1.],
		[0.0005*tsh/1., 1-0.0005*tsh/1.]])

	tmatrix = np.array([
		[t1[0][0]*t3[0][0], t1[0][1]*t3[0][0], t2[0][0]*t3[0][1], t2[0][1]*t3[0][1]],
		[t1[1][0]*t3[0][0], t1[1][1]*t3[0][0], t2[1][0]*t3[0][1], t2[1][1]*t3[0][1]],
		[t1[0][0]*t3[1][0], t1[0][1]*t3[1][0], t2[0][0]*t3[1][1], t2[0][1]*t3[1][1]],
		[t1[1][0]*t3[1][0], t1[1][1]*t3[1][0], t2[1][0]*t3[1][1], t2[1][1]*t3[1][1]]]) 

	tmatrix = None 
	
	set_init = (mu, var, ppi, tmatrix)

	new_d = []
	for i in range(len(dataset)):
		yi = dataset[i].astype('double')
		xn = np.where(np.isfinite(yi))[0]
		yi = yi[xn]
		new_d.append(yi)

	from tmaven.controllers.modeler.hmm_vb_consensus import consensus_vb_em_hmm
	result = consensus_vb_em_hmm(new_d,nstates,maxiters=maxiters,threshold=converge,nrestarts=nrestarts,priors=priors, mu_mode=True, set_init=set_init)

	return result

def idealize_consensusHMM(data,result):
	from tmaven.controllers.modeler.fxns.hmm import viterbi
	result.idealized = np.zeros_like(data) + np.nan
	result.chain = np.zeros_like(data) + np.nan
	for i in range(data.shape[0]):
		result.chain[i] = viterbi(data[i],result.mean,result.var,result.tmatrix,result.frac).astype('int')
		result.idealized[i] = result.mean[viterbi(data[i],result.mean,result.var,result.tmatrix,result.frac).astype('int')]

	return result

def analyze_consensusHMM(dataset, nstates):
	result = run_consensusHMM(dataset, nstates)
	result = idealize_consensusHMM(dataset, result)

	return result
