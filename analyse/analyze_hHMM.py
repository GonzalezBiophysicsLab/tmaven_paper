import numpy as np

def run_hHMM(dataset, depth_vec, prod_states):
    max_iter = 100
    tol = 1e-4
    restarts = 2
    guess = np.array([0., 1.])
    k = prod_states

    from tmaven.controllers.modeler.hhmm_vmp import vmp_hhmm
    post_mu, post_beta, post_a, post_b, post_pi, post_tm, post_exit,likelihood = vmp_hhmm(dataset,
                                                                           depth_vec,prod_states,
                                                                           max_iter,tol,
                                                                           restarts,guess)

    #hardcoded to 4 state
    flat_tm = np.zeros((4,4),dtype=np.float64)
    #result = [post_mu, post_beta, post_a, post_b, post_pi, post_tm, post_exit]

    new_tm = []
    for level in post_tm:
        new_level = []
        for tm in level:
            newt = tm.T.copy()
            for i in range(newt.shape[0]):
                newt[i] /= newt[i].sum()
            new_level.append(newt)
        new_tm.append(new_level)

    flat_tm[:2,:2] = new_tm[1][0]
    flat_tm[-2:,-2:] = new_tm[1][1]
    flat_tm[:2,-2:] = post_exit[1][0].reshape(2,1)@post_pi[1][1].reshape(1,2)*new_tm[0][0][0,1]
    flat_tm[-2:,:2] = post_exit[1][1].reshape(2,1)@post_pi[1][0].reshape(1,2)*new_tm[0][0][1,0]

    #print(flat_tm)
    mu = np.zeros(4)
    mu[:2] = post_mu.flatten()
    mu[-2:] = post_mu.flatten()
    var = np.zeros(4)
    var[:2] = (post_b/post_a).flatten()
    var[-2:] = (post_b/post_a).flatten()

    upper_pi = (post_exit[0][0]*post_pi[0][0]).flatten()
    pi =  np.zeros(4)
    pi[:2] = upper_pi*post_pi[1][0]*post_exit[1][1].flatten()
    pi[-2:] = upper_pi*post_pi[1][1]*post_exit[1][0].flatten()
    pi /= pi.sum()

    for i in range(flat_tm.shape[0]):
        flat_tm[i] /= flat_tm[i].sum()

    from tmaven.controllers.modeler.model_container import model_container

    result = model_container(type='hierarchical HMM',
						  nstates = 4,mean=mu,var=var,frac=pi,
						  tmatrix=flat_tm,
						  likelihood=likelihood,
						  a=post_a,b=post_b,beta=post_beta, h_pi=post_pi, h_tm=post_tm, h_exit=post_exit)

    return result

def analyze_hHMM(dataset, *args):
    depth_vec = [2]
    prod_states = 2

    result = run_hHMM(dataset, depth_vec, prod_states)
    #idealise

    return result
