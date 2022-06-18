import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from generate_trace import generate_chain, add_emission

def simulate_dynamic2(rep_number,nrestarts,nmols,nt):
    mu =  np.array([0.05, 0.75, 0.95, 0.05, 0.95])
    s =  np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    pi = np.array([0.2, 0.15, 0.1, 0.6, 0.3])



    transition = np.array([[0.980, 0.010, 0.010, 0.008/6, 0.008/6],
                           [0.015, 0.970, 0.015, 0.008/6, 0.008/6],
                           [0.020, 0.020, 0.960, 0.008/6, 0.008/6],
                        [0.008/18, 0.008/18, 0.008/18, 0.98, 0.02],
                        [0.008/18, 0.008/18, 0.008/18, 0.04, 0.96]])


    traces = []
    vits = []
    chains = []

    N = nmols
    T = nt
    K = np.arange(len(mu), dtype = 'float64')

    dataset_number = 3

    seed = nrestarts*dataset_number+rep_number
    for j in range(N):

        c =  generate_chain(T,K,pi, transition, 666 + seed + j)
        i,t = add_emission(c,K,mu,s)

        chains.append(c)
        traces.append(t)
        vits.append(i)


    chains = np.array(chains)
    traces = np.array(traces)
    vits = np.array(vits)

    return traces, vits, chains

if __name__ == '__main__':

    traces,vits,chains = simulate_dynamic2(1,1,200,1000)
    print (traces[100][0:10])
    plt.plot(traces[100])
    #plt.plot(vits[20], 'k')
    plt.plot(chains[100], 'k')
    plt.show()


    plt.hist(np.concatenate(traces),bins = 100)
    plt.show()
