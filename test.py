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

for snr in [9.0, 3.0, 6.0]:
	print(snr)
	simulations = [simulate_reg,simulate_static2,simulate_dynamic2,simulate_static3,simulate_dynamic3]
	dataset = ['reg','static2','dynamic2','static3','dynamic3']

	analysis = [analyze_consensusHMM, analyze_ebHMM, analyze_vbHMM_GMM]
	model = ['consensus', 'eb', 'vbHMM+GMM']

	nstates = 2

	nrestarts = 100
	prop = np.array([0.25,0.75])

	for j in range(3):
		sim_dataset = simulations[0]
		ds = dataset[0]
		print(ds)
		analyse = analysis[j]
		model_used = model[j]
		print(model_used)
		Es = []
		for i in tqdm(range(nrestarts)):
			traces,vits,chains= sim_dataset(i,nrestarts,200,1000,snr)

			res = analyse(traces, nstates)
			#print(res.mean, res.var, res.frac, res.tmatrix)
			res.tmstar = res.tmatrix.copy()
			for i in range(res.tmstar.shape[0]):
				res.tmstar[i] /= res.tmstar[i].sum()

			t_res,E_y0yt_res = gen_mc_acf(1,100,res.tmstar,res.mean,res.var,res.frac)
			Es.append(E_y0yt_res)

		#print(res.mean)
		#print(res.vb_means)
		ylims = (-0.2, 0.4)
		plot_acf(t_res, Es, ds, 'TM', '{}({})'.format(model_used,nstates), snr, pb='NoPB', prop=prop, ylims=ylims)
		plot_acf_residuals(t_res, Es, ds, 'TM', '{}({})'.format(model_used,nstates), snr, pb='NoPB', prop=prop, ylims=ylims)




#print(res.tmatrix/(res.tmatrix.sum(1)[:,None]))
