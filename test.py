from simulate.simulate_reg import simulate_reg
from analyse.analyze_consensusHMM import analyze_consensusHMM
from analyse.acf import gen_mc_acf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


reg_mu =  np.array([0.05, 0.95])
reg_s =  np.array([0.1, 0.1])
reg_pi = np.array([0.6, 0.4])
reg_transition = np.array([[0.98, 0.02],
						   [0.03, 0.97]])

t, E_y0yt = gen_mc_acf(1,100,reg_transition,reg_mu,reg_s**2,reg_pi)
plt.plot(t, E_y0yt, 'r')

for i in tqdm(range(10)):
	traces,vits,chains= simulate_reg(i,1,200,1000)

	'''
	print("Generated")
	print (traces[20][0:10])
	plt.plot(traces[20])
	plt.show()

	plt.plot(chains[20], 'k')
	plt.show()

	plt.hist(np.concatenate(traces),bins = 100)
	plt.show()
	'''

	res = analyze_consensusHMM(traces, 2)
	#print(res.mean, res.var, res.frac, res.tmatrix)
	res.tmstar = res.tmatrix.copy()
	for i in range(res.tmstar.shape[0]):
		res.tmstar[i] /= res.tmstar[i].sum()
		#print(res.tmstar)
		#print("Analysed")


	t_res,E_y0yt_res = gen_mc_acf(1,100,res.tmstar,res.mean,res.var,res.frac)
	plt.plot(t_res, E_y0yt_res, alpha = 0.1)

plt.show()


#print(res.tmatrix/(res.tmatrix.sum(1)[:,None]))
