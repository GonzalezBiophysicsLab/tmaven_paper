from simulate.simulate_hom import simulate_hom
from simulate.simulate_hetstat import simulate_hetstat
from simulate.simulate_hetdyn import simulate_hetdyn

from analyse.analyze_consensusHMM import analyze_consensusHMM
from analyse.analyze_vbHMM_GMM import analyze_vbHMM_GMM
from analyse.analyze_hHMM import analyze_hHMM
from analyse.acf import gen_mc_acf

from plot.acf_plot import plot_acf, plot_acf_residuals
from tqdm import tqdm
import pickle
import os


def main():
	import argparse
	parser = argparse.ArgumentParser(description="Run the tMAVEN paper calculations")
	parser.add_argument('dataset', type=str, choices=['hom','hetstat','hetdyn'],help='which dataset to simulate')
	parser.add_argument('change', type=str, choices=['fixed', 'changeN','changeT','changeprop','changerate'], help='what to change')
	parser.add_argument('model', type=str, choices=['composite','global','hhmm'],help='which model to use for analysis')
	args = parser.parse_args()

	# Default parameters for simulating datasets
	nstates = 2
	snr = 9.
	T = 1000
	N = 200
	nrestarts = 100
	prop = 0.25
	tsh = 4
	index = 'Fixed'

	# Setting up simulations
	simulations = {'hom':simulate_hom,'hetstat':simulate_hetstat,'hetdyn':simulate_hetdyn}
	dataset = args.dataset
	simulate = simulations[dataset]
	print("Simulating -- {}".format(dataset))

	# Setting up what to change
	change = args.change
	print("Changing -- {}".format(change))
	if change == 'fixed':
		sim_range = 1
	elif dataset == 'hom' and change == 'changeT':
		sim_range=4
		Ts = [100,300,1000,3000]
	elif dataset == 'hom' and change == 'changeN':
		sim_range = 3
		Ns = [100,200,500]
	elif dataset == 'hetstat' and change == 'changeprop':
			sim_range = 20
	elif dataset == 'hetdyn' and change == 'changerate':
			sim_range = 20
	else:
		print("Invalid combination!!")
		return
	
	# Setting up analysis
	analysis = {'composite': analyze_vbHMM_GMM, 'global': analyze_consensusHMM,'hhmm': analyze_hHMM}
	model = args.model
	analyse = analysis[model]
	print('Analyzing -- {}'.format(model))

	Edict = {}
	resdict = {}

	for j in range(sim_range):
		if change == 'changeN':
			N = Ns[j]
			index = N
			print("N = {}".format(N))
		elif change == 'changeT':
			T = Ts[j]
			index = T
			print("T= {}".format(T))
		elif change == 'changeprop':
			prop = 0.05*(j+1) # proportion of fast. ONLY USED FOR STAT HET
			index = prop
			print("prop = {}".format(prop))
		elif change == 'changerate':
			tsh = j+1 # metric of transition between slow and fast. ONLY USED FOR DYN HET
			index = tsh
			print("tsh = {}".format(tsh))

		ress = []
		Es = []

		for i in tqdm(range(nrestarts)):
			traces,vits,chains= simulate(i,nrestarts,N,T,snr, prop=prop, tsh=tsh) #these functions are defined so that prop is only used for stat-het and tsh for dyn-het

			res = analyse(traces, nstates)

			res.tmstar = res.tmatrix.copy()
			for i in range(res.tmstar.shape[0]):
				res.tmstar[i] /= res.tmstar[i].sum()

			t_res,E_y0yt_res = gen_mc_acf(1,1000,res.tmstar,res.mean,res.var,res.frac)
			Es.append(E_y0yt_res)
			ress.append(res)
		
		Edict[index] = Es
		resdict[index] = ress

	if not os.path.exists('figures/results'):
		os.makedirs('figures/results')

	pickle.dump(Edict, open( "figures/results/ACFdicts_{}_{}_{}_{}_{}.p".format(nstates,dataset,snr,model,change), "wb" ))
	pickle.dump(resdict, open( "figures/results/Resultdicts_{}_{}_{}_{}_{}.p".format(nstates,dataset,snr,model,change), "wb" ) )

if __name__=='__main__':
	main()