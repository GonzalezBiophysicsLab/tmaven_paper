import numpy as np
from simulate.simulate_hom import simulate_reg
from simulate.simulate_hetstat import simulate_static2
from simulate.simulate_hetdyn import simulate_dynamic2

simulations = [simulate_reg,simulate_static2,simulate_dynamic2]

#### actual calls to tMAVEN functions go here
def analyze_vbFRET_GMM(dataset,nstates):
	traces, vits, chains = dataset
	analysis = np.random.rand()
	return analysis

def analyze_consensusHMM(dataset,nstates):
	traces, vits, chains = dataset
	analysis = np.random.rand()
	return analysis

def analyze_hFRET(dataset,nstates):
	traces, vits, chains = dataset
	analysis = np.random.rand()
	return analysis

#### Analyze a replicate and save results
def run_vbfret(theta):
	global simulations
	rep,nrestarts,nmols,nt,snrs,truncs = theta

	out = {}
	for snr in snrs:
		for trunc in truncs:
			for sim_ind, sim in zip(range(len(simulations)),simulations):
				dataset = sim(rep,nrestarts,nmols,nt,snr,trunc)
				out['%d %d %.2f %s'%(rep,sim_ind,snr,str(trunc))] = analyze_vbFRET_GMM(dataset,2)
	import pickle
	with open('results/vbfret_%d.pickle'%(rep),'wb') as f:
		pickle.dump(out,f)

def run_consensus(theta):
	global simulations
	rep,nrestarts,nmols,nt,snrs,truncs = theta

	out = {}
	for snr in snrs:
		for trunc in truncs:
			for sim_ind, sim in zip(range(len(simulations)),simulations):
				dataset = sim(rep,nrestarts,nmols,nt,snr,trunc)
				out['%d %d %.2f %s'%(rep,sim_ind,snr,str(trunc))] = analyze_consensusHMM(dataset,2)
	import pickle
	with open('results/consensus_%d.pickle'%(rep),'wb') as f:
		pickle.dump(out,f)

def run_hfret(theta):
	global simulations
	rep,nrestarts,nmols,nt,snrs,truncs = theta

	out = {}
	for snr in snrs:
		for trunc in truncs:
			for sim_ind, sim in zip(range(len(simulations)),simulations):
				dataset = sim(rep,nrestarts,nmols,nt,snr,trunc)
				out['%d %d %.2f %s'%(rep,sim_ind,snr,str(trunc))] = analyze_hFRET(dataset,2)
	import pickle
	with open('results/hfret_%d.pickle'%(rep),'wb') as f:
		pickle.dump(out,f)


#### Do analysis things
def run_analysis():
	print('running analysis')
	import multiprocessing

	nrestarts = 100
	nmols = 200
	nt = 1000
	snrs = [3.,6.,9.]
	truncs = [None,500.]

	import os
	if not os.path.isdir('results'):
		os.mkdir('results')

	with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
		thetas = [[i,nrestarts,nmols,nt,snrs,truncs] for i in range(nrestarts)]
		pool.map(run_vbfret,thetas)
		pool.map(run_consensus,thetas)
		pool.map(run_hfret,thetas)

def collect_results():
	print('collecting results')
	# ''' no input'''
	# fname = 'important_output_filename.hdf5'
	# import h5py
	# global nrestarts
	# out = {'vb':{},'con':{},'hfret':{}}
	# for i in nrestarts:
	# 	out['vb']['%d'] = load_restart_datasettype_vb...
	# 	out['con']['%d'] = load_restart_datasettype_consensus...
	# 	out['hfret']['%d'] = load_restart_datasettype_hfret...
	# with h5py.File(fname,'w') as f:
	# 	... save results ....

def make_figures():
	print('making figures!')
	# ''' no input'''
	#
	# ...load results...
	#
	# ...do stuff...
	#
	# ...make/save figures...


def main():
	import argparse
	parser = argparse.ArgumentParser(description="Run the tMAVEN paper calculations")
	parser.add_argument('step', type=str, choices=['run','collect','figures'],help='which step to run')
	args = parser.parse_args()
	if args.step == 'run':
		run_analysis()
	elif args.step ==  'collect':
		collect_results()
	elif args.step == 'figures':
		make_figures()

if __name__ == '__main__':
	main()
