from simulate.simulate_reg import simulate_reg
from simulate.simulate_static2 import simulate_static2
from simulate.simulate_static3 import simulate_static3
from simulate.simulate_dynamic2 import simulate_dynamic2
from simulate.simulate_dynamic3 import simulate_dynamic3

from analyse.analyze_consensusHMM import analyze_consensusHMM
from analyse.analyze_ebHMM import analyze_ebHMM
from analyse.acf import gen_mc_acf

from plot.acf_plot import plot_acf, plot_acf_residuals
import numpy as np
from tqdm import tqdm

snr = 9.
simulations = [simulate_reg,simulate_static2,simulate_dynamic2,simulate_static3,simulate_dynamic3]
dataset = ['reg','static2','dynamic2','static3','dynamic3']

nrestarts = 100
prop = np.array([0.25,0.75])

for i in range(5):
	sim_dataset = simulations[i]
	ds = dataset[i]
	print(ds)
	Es = []
	for i in tqdm(range(nrestarts)):
		traces,vits,chains= sim_dataset(i,nrestarts,200,1000,snr)

		res = analyze_ebHMM(traces, 3)
		#print(res.mean, res.var, res.frac, res.tmatrix)
		res.tmstar = res.tmatrix.copy()
		for i in range(res.tmstar.shape[0]):
			res.tmstar[i] /= res.tmstar[i].sum()
			#print(res.tmstar)
			#print("Analysed")


		t_res,E_y0yt_res = gen_mc_acf(1,100,res.tmstar,res.mean,res.var,res.frac)
		Es.append(E_y0yt_res)

	plot_acf(t_res, Es, ds, 'TM', 'eb', snr, pb='NoPB', prop=prop)
	plot_acf_residuals(t_res, Es, ds, 'TM', 'eb', snr, pb='NoPB', prop=prop)




#print(res.tmatrix/(res.tmatrix.sum(1)[:,None]))
