import numpy as np

def run_mlHMM_GMM(dataset, nstates):
	'''
	'modeler.nrestarts':10,
	'modeler.converge':1e-10,
	'modeler.maxiters':1000,
	'''

	maxiters = 1000
	nrestarts = 10
	converge = 1e-10
	from tmaven.controllers.modeler.fxns.hmm import viterbi

	idealized = np.zeros_like(dataset) + np.nan
	ml_results = []

	for i in range(len(dataset)):
		yi = dataset[i].astype('double')
		xn = np.where(np.isfinite(yi))[0]
		yi = yi[xn]

		from tmaven.controllers.modeler.hmm_ml import ml_em_hmm
		r = ml_em_hmm(yi,nstates,maxiters=maxiters,threshold=converge,init_kmeans=True)

		vit = r.mean[viterbi(yi,r.mean,r.var,r.tmatrix,r.frac).astype('int')]
		idealized[i,xn] = vit
		ml_results.append(r)

	vits = np.concatenate(idealized)
	vits = vits[np.isfinite(vits)]
	priors = np.array([0.25,2.5,0.01,1.])

	from tmaven.controllers.modeler.gmm_vb import vb_em_gmm

	result = vb_em_gmm(vits,nstates,maxiters=maxiters,prior_strengths=priors,threshold=converge,init_kmeans=True)

	idealized_gmm = np.zeros_like(dataset) + np.nan
	chain = np.zeros_like(dataset, dtype = 'int')
	ml_means = []
	ml_vars = []
	ml_tmatrices = []
	ml_frac = []
	tmatrix = np.zeros((nstates,nstates))
	varsum = np.zeros(nstates)
	vardenom = np.zeros(nstates)

	for i in range(len(dataset)):
		y = dataset[i]
		ml = ml_results[i]

		prob = 1./np.sqrt(2.*np.pi*result.var[None,None,:])*np.exp(-.5/result.var[None,None,:]*(idealized[i,:,None]-result.mean[None,None,:])**2.)
		prob /= prob.sum(2)[:,:,None]
		prob *= result.frac[None,None,:]
		idealpath = np.argmax(prob,axis=2).astype('int64')
		idealized_gmm[i] = result.mean[idealpath]
		chain[i] = idealpath

		ml_means.append(ml.mean)
		ml_vars.append(ml.var)
		ml_tmatrices.append(ml.tmatrix)
		ml_frac.append(ml.frac)
		'''
		if len(ml.var) == nstates:
			tmatrix += ml.tmatrix
			varsum += ml.var
			vardenom += 1		print(probs)
		else:			for i in range(res.tmstar.shape[0]):
				res.tmstar[i] /= res.tmstar[i].sum()
			state_set = set(chain[i])

			for j,k in enumerate(state_set):
				varsum[k] += ml.var[j]
				vardenom[k] += 1
				for m,n in enumerate(state_set):
					tmatrix[k,n] += ml.tmatrix[j,m]
		'''
		probs = 1./np.sqrt(2.*np.pi*result.var[None,:])*np.exp(-.5/result.var[None,:]*(ml.mean[:,None]-result.mean[None,:])**2.)
		probs /= probs.sum(1)[:,None]
		#print(probs)
		tmatrix += ml.tmatrix
		state_set = np.sort(np.unique(chain[i]))
		for j,k in enumerate(state_set):
			varsum[k] += (ml.var*probs[:,j]).sum()
			vardenom[k] += probs[:,j].sum()
			#for m,n in enumerate(state_set):

				#tmatrix[k,n] += (ml.tmatrix*(probs[:,j][:,None])*(probs[:,m][None,:])).sum()

	var = varsum/vardenom
	#print(var, tmatrix)
	tmatrix /= len(dataset)
	viterbi_var = result.var
	y_flat = np.concatenate(dataset)
	y_flat = y_flat[np.isfinite(y_flat)]
	#softmax_var = ((result.r*(y_flat[:,None] - result.mean[None,:])**2)).sum(0)/(result.r).sum(0)

	softmax_var = np.zeros_like(result.mean)
	for i,state in enumerate(np.argmax(result.r, axis = 1)):
		softmax_var[state] += (y_flat[i] - result.mean[state])**2

	softmax_var /= result.r.sum()
	#print(viterbi_var,var)
	result.var = var
	result.softmax_var = softmax_var
	result.viterbi_var = viterbi_var
	result.idealized_gmm = idealized_gmm
	result.idealized_hmm = idealized
	result.ml_means = ml_means
	result.ml_vars = ml_vars
	result.ml_tmatrices = ml_tmatrices
	result.ml_frac = ml_frac
	result.tmatrix = tmatrix
	result.chain = chain

	return result

def analyze_mlHMM_GMM(dataset, nstates):
	result = run_mlHMM_GMM(dataset, nstates)

	return result
