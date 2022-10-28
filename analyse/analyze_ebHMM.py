import numpy as np

def run_ebHMM(dataset, nstates):

	'''
	'modeler.nrestarts':10,
	'modeler.converge':1e-6,
	'modeler.maxiters':20,
	'modeler.ebhmm.prior.beta':0.25,
	'modeler.ebhmm.prior.a':2.5,
	'modeler.ebhmm.prior.b':0.01,
	'modeler.ebhmm.prior.alpha':1.,
	'modeler.ebhmm.prior.pi':1.
	'''

	maxiters = 100
	nrestarts = 10
	converge = 1e-6
	new_d = []
	for i in range(len(dataset)):
		yi = dataset[i].astype('double')
		xn = np.where(np.isfinite(yi))[0]
		yi = yi[xn]
		new_d.append(yi)

	mu_prior = np.percentile(np.concatenate(new_d),np.linspace(0,100,nstates+2))[1:-1]
	beta_prior = np.ones_like(mu_prior)*0.25
	a_prior = np.ones_like(mu_prior)*2.5
	b_prior = np.ones_like(mu_prior)*0.01
	pi_prior = np.ones_like(mu_prior)*1.
	tm_prior = np.ones((nstates,nstates))*1.

	priors = [mu_prior, beta_prior, a_prior, b_prior, pi_prior, tm_prior]

	from tmaven.controllers.modeler.hmm_eb import eb_em_hmm
	result, vbresults = eb_em_hmm(new_d,nstates,maxiters=maxiters,threshold=converge,
								  nrestarts=nrestarts,priors=priors,init_kmeans=True)

	return result, vbresults

def idealize_ebHMM(data,result,vbresults):
	from tmaven.controllers.modeler.fxns.hmm import viterbi

	idealized = np.zeros_like(data) + np.nan
	chain = np.zeros_like(data, dtype = 'int')
	vb_means = []
	vb_vars = []
	vb_tmatrices = []

	for i in range(len(data)):
		y = data[i]
		vb = vbresults[i]

		vb_means.append(vb.mean)
		vb_vars.append(vb.var)
		vb_tmatrices.append(vb.tmatrix)

		chain[i] = viterbi(y,vb.mean,vb.var,vb.tmatrix,vb.frac).astype('int')
		idealized[i] = vb.mean[chain[i]]


	result.vb_means = vb_means
	result.vb_vars = vb_vars
	result.vb_tmatrices = vb_tmatrices
	result.chain = chain
	result.idealized = idealized

	return result


def analyze_ebHMM(dataset, nstates):
	result, vbs = run_ebHMM(dataset, nstates)
	result = idealize_ebHMM(dataset, result, vbs)

	return result
