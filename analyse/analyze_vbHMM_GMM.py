import numpy as np

def run_vbHMM_GMM(dataset, nstates):
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

	maxiters = 20
	nrestarts = 10
	converge = 1e-6
	from tmaven.controllers.modeler.fxns.hmm import viterbi

	idealized = np.zeros_like(dataset) + np.nan
	vb_results = []
	for i in range(len(dataset)):
		yi = dataset[i].astype('double')
		results = []
		for k in range(1,nstates+1):
			mu_prior = np.percentile(np.concatenate(dataset),np.linspace(0,100,k+2))[1:-1]
			beta_prior = np.ones_like(mu_prior)*0.25
			a_prior = np.ones_like(mu_prior)*2.5
			b_prior = np.ones_like(mu_prior)*0.01
			pi_prior = np.ones_like(mu_prior)*1.
			tm_prior = np.ones((k,k))*1.

			priors = [mu_prior, beta_prior, a_prior, b_prior, pi_prior, tm_prior]
			from tmaven.controllers.modeler.hmm_vb import vb_em_hmm
			results.append(vb_em_hmm(yi,k,maxiters=maxiters,threshold=converge,priors=priors,init_kmeans=True))

		elbos = np.array([ri.likelihood[-1,0] for ri in results])
		modelmax = np.argmax(elbos)
		r = results[modelmax]

		vit = r.mean[viterbi(yi,r.mean,r.var,r.tmatrix,r.frac).astype('int')]
		idealized[i] = vit
		vb_results.append(r)

	vits = np.concatenate(idealized)
	vits = vits[np.isfinite(vits)]
	priors = np.array([0.25,2.5,0.01,1.])

	from tmaven.controllers.modeler.gmm_vb import vb_em_gmm

	result = vb_em_gmm(vits,nstates,maxiters=maxiters,threshold=converge,prior_strengths=priors,init_kmeans=True)

	idealized_gmm = np.zeros_like(dataset) + np.nan
	chain = np.zeros_like(dataset, dtype = 'int')
	vb_means = []
	vb_vars = []
	vb_tmatrices = []
	vb_frac = []
	tmatrix = np.zeros((nstates,nstates))

	for i in range(len(dataset)):
		y = dataset[i]
		vb = vb_results[i]


		vb_means.append(vb.mean)
		vb_vars.append(vb.var)
		vb_tmatrices.append(vb.tmatrix)
		vb_frac.append(vb.frac)
		tmatrix += vb.tmatrix

		prob = 1./np.sqrt(2.*np.pi*result.var[None,None,:])*np.exp(-.5/result.var[None,None,:]*(idealized[i,:,None]-result.mean[None,None,:])**2.)
		prob /= prob.sum(2)[:,:,None]
		prob *= result.frac[None,None,:]
		idealpath = np.argmax(prob,axis=2).astype('int64')
		idealized_gmm[i] = result.mean[idealpath]
		chain[i] = idealpath

	viterbi_var = result.var
	#var = (result.r*((y_flat[:,None] - result.mean[None,:])**2)).sum(0)/(result.r).sum()
	var = np.zeros_like(result.mean)
	y_flat = np.concatenate(dataset)
	for i,state in enumerate(np.argmax(result.r, axis = 1)):
		var[state] += (y_flat[i] - result.mean[state])**2

	var /= result.r.sum()

	#print(viterbi_var,var)
	result.var = var
	result.viterbi_var = viterbi_var
	result.idealized_gmm = idealized_gmm
	result.idealized_hmm = idealized
	result.vb_means = vb_means
	result.vb_vars = vb_vars
	result.vb_tmatrices = vb_tmatrices
	result.vb_frac = vb_frac
	result.tmatrix = tmatrix
	result.chain = chain

	return result

def analyze_vbHMM_GMM(dataset, nstates):
	result = run_vbHMM_GMM(dataset, nstates)

	return result
