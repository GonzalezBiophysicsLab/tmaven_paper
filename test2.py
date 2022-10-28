from simulate.simulate_reg import simulate_reg
from simulate.simulate_static2 import simulate_static2
from simulate.simulate_static3 import simulate_static3
from simulate.simulate_dynamic2 import simulate_dynamic2
from simulate.simulate_dynamic3 import simulate_dynamic3

from analyse.analyze_consensusHMM import analyze_consensusHMM
from analyse.analyze_ebHMM import analyze_ebHMM
from analyse.analyze_vbHMM_GMM import analyze_vbHMM_GMM
from analyse.acf import gen_mc_acf

# from plot.acf_plot import plot_acf, plot_acf_residuals
import numpy as np
# from tqdm import tqdm
# import pickle

print('imported')


runs = np.arange(2)
for i in runs:
	# traces,vits,chains= simulate_reg(i,runs.size,20,1000,9.0,truncate=200.)
	traces,vits,chains= simulate_reg(i,runs.size,20,1000,9.0,truncate=200.)

	# print(traces.shape,traces.dtype)
	# print('simulated')
	res = analyze_ebHMM(traces, 2)

	print(res.mean, res.var, res.frac)

	res = analyze_vbHMM_GMM(traces, 2)
	print(res.mean, res.var, res.frac)
	print('\n')
