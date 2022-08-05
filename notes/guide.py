import tmaven

nrestarts = 100
nmol = 200
nt = 1000
SNRS = [3.,6.,9.]
truncs = [True,False]



def seed(dataset_number,rep_number):
    ''' rep_number in [0,nrestarts) gives the EXACT same dataset every time'''
    global nrestarts
    np.random.seed(666+nrestarts*dataset_number+rep_number)

def simulate_reg(rep_number,SNR,flag_truncate):
    global nmol,nt
    seed(0,rep_number)
    dataset = ...
    return dataset

def simulate_static2(rep_number,SNR,flag_truncate):
    global nmol,nt
    seed(1,rep_number)
    dataset = ...
    return dataset

def simulate_dynamic2(rep_number,SNR,flag_truncate):
    global nmol,nt
    seed(2,rep_number)
    dataset = ...
    return dataset

def simulate_static3(rep_number,SNR,flag_truncate):
    global nmol,nt
    seed(3,rep_number)
    dataset = ...
    return dataset

def simulate_dynamic3(rep_number,SNR,flag_truncate):
    global nmol,nt
    seed(4,rep_number)
    dataset = ...
    return dataset

simulations = [simulate_reg,simulate_static2,simulate_dynamic2,simulate_static3,simulate_dynamic3]

def analyze_vbFRET_GMM(dataset,nstates):
    ...
    return analysis

def analyze_consensusHMM(dataset,nstates):
    ...
    return analysis

def analyze_hFRET(dataset,nstates,...):
    ...
    return analysis

def run_vb(rep):
    global SNRS,truncs,simulations

    out = {}
    for SNR in SNRS:
        for trunc in truncs:
            for sim_ind, sim in zip(range(len(simulations)),simulations):
                dataset = sim(rep,SNR,trunc)
                out['%d %d'%(rep,sim_ind)] = analyze_vbFRET_GMM(dataset,2)
    ...save out to file...

def run_con(rep):
    global SNRS,truncs,simulations

    out = {}
    for SNR in SNRS:
        for trunc in truncs:
            for sim_ind, sim in zip(range(len(simulations)),simulations):
                dataset = sim(rep,SNR,trunc)
                out['%d %d'%(rep,sim_ind)] = analyze_consensusHMM(dataset,2)
    ...save out to file...

def run_hfret(rep):
    global SNRS,truncs,simulations

    out = {}
    for SNR in SNRS:
        for trunc in truncs:
            for sim_ind, sim in zip(range(len(simulations)),simulations):
                dataset = sim(rep,SNR,trunc)
                out['%d %d'%(rep,sim_ind)] = analyze_hFRET(dataset,2,...)
    ...save out to file...

def run_analysis():
    ''' no input '''
    import multiprocessing

    pool = multiprocessing.Pool(ncpu=8)
    pool.map(run_vb,np.arange(nrestarts))
    pool.map(run_con,np.arange(nrestarts))
    pool.map(run_hfret,np.arange(nrestarts))

def collect_results():
    ''' no input'''
    fname = 'important_output_filename.hdf5'
    import h5py
    global nrestarts
    out = {'vb':{},'con':{},'hfret':{}}
    for i in nrestarts:
        out['vb']['%d'] = load_restart_datasettype_vb...
        out['con']['%d'] = load_restart_datasettype_consensus...
        out['hfret']['%d'] = load_restart_datasettype_hfret...
    with h5py.File(fname,'w') as f:
        ... save results ....

def make_figures():
    ''' no input'''

    ...load results...

    ...do stuff...

    ...make/save figures...

def __main__():
    import argparse
    parser = argparse.ArgumentParser(description="Run the tMAVEN paper calculations")
	parser.add_argument('step', type=str, choices=['model','collect','figures'],help='which step to run'
    args.parser.parse_args()
    if args.step == 'model':
        run_analysis()
    elif args.step ==  'collect':
        collect_results()
    elif args.step == 'figures':
        make_figures()

if __name__ == '__main__':
    main()
