from simulate.simulate_reg import simulate_reg
from simulate.simulate_static2 import simulate_static2
from simulate.simulate_static3 import simulate_static3
from simulate.simulate_dynamic2 import simulate_dynamic2
from simulate.simulate_dynamic3 import simulate_dynamic3

from analyse.analyze_consensusHMM import analyze_consensusHMM
from analyse.analyze_ebHMM import analyze_ebHMM
from analyse.analyze_vbHMM_GMM import analyze_vbHMM_GMM
from analyse.acf import gen_mc_acf

from plot.acf_plot import plot_acf, plot_acf_residuals
import numpy as np
from tqdm import tqdm
import pickle

for snr in [9.0]:#, 3.0, 6.0]:
	print(snr)
	simulations = [simulate_reg,simulate_static2,simulate_dynamic2,simulate_static3,simulate_dynamic3]
	dataset = ['reg','static2','dynamic2','static3','dynamic3']

	analysis = [ analyze_vbHMM_GMM, analyze_consensusHMM, analyze_ebHMM]
	model = ['vbHMM+GMM','consensus', 'eb']

	nstates = 2

	nrestarts = 100
	prop = np.array([0.25,0.75])
	Edict = {}
	resdict = {}
	for j in range(10):
		tau = 1000
		N = 200
		truncate = 50*(j+1)
		sim_dataset = simulations[0]
		ds = dataset[0]
		print(ds)
		analyse = analysis[2]
		model_used = model[2]
		print(model_used)
		print(truncate)
		ress = []
		Es = []
		for i in tqdm(range(nrestarts)):
			traces,vits,chains= sim_dataset(i,nrestarts,N,tau,snr,truncate=truncate)

			res = analyse(traces, nstates)
			#print(res.mean, res.var, res.frac, res.tmatrix)
			res.tmstar = res.tmatrix.copy()
			for i in range(res.tmstar.shape[0]):
				res.tmstar[i] /= res.tmstar[i].sum()
			#print(res.var)

			t_res,E_y0yt_res = gen_mc_acf(1,1000,res.tmstar,res.mean,res.var,res.frac)
			Es.append(E_y0yt_res)
			ress.append(res)
		Edict[truncate] = Es
		resdict[truncate] = ress

		pickle.dump(Edict, open( "ACFdicts_{}_{}_{}_{}_PB.p".format(nstates,ds,snr,model_used), "wb" ))
		pickle.dump(resdict, open( "Resultdicts_{}_{}_{}_{}_PB.p".format(nstates,ds,snr,model_used), "wb" ) )
		#print(res.mean)
		#print(res.vb_means)
		#ylims = (-0.02, 0.04)
		#plot_acf(t_res, Es, ds, 'TM', 'new{}({})'.format(model_used,nstates), snr, pb='NoPB', prop=prop)
		#plot_acf_residuals(t_res, Es, ds, 'TM', 'new{}({})'.format(model_used,nstates), snr, pb='NoPB', prop=prop, ylims=ylims)




#print(res.tmatrix/(res.tmatrix.sum(1)[:,None]))
