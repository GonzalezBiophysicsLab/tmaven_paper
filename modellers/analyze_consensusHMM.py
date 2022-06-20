import numpy as np

def analyze_consensusHMM(dataset, nstates):

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

    mu_prior = np.percentile(np.concatenate(dataset),np.linspace(0,100,nstates+2))[1:-1]
    beta_prior = np.ones_like(mu_prior)*0.25
    a_prior = np.ones_like(mu_prior)*2.5
    b_prior = np.ones_like(mu_prior)*0.01
    pi_prior = np.ones_like(mu_prior)*1.
    tm_prior = np.ones((nstates,nstates))*1.

    priors = [mu_prior, beta_prior, a_prior, b_prior, pi_prior, tm_prior]

    maxiters = 1000
    nrestarts = 4
    converge = 1e-10

    from tmaven.controllers.modeler.hmm_vb_consensus import consensus_vb_em_hmm
    result = consensus_vb_em_hmm(dataset,nstates,maxiters=maxiters,threshold=converge,nrestarts=nrestarts,priors=priors,init_kmeans=True)

    return result
