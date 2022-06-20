import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from .generate_trace import generate_chain, add_emission

def simulate_static3(rep_number,nrestarts,nmols,nt):
    mu1 =  np.array([0.05, 0.75, 0.95])
    s1 =  np.array([0.1, 0.1, 0.1])
    pi1 = np.array([0.6, 0.45, 0.3])

    transition1 = np.array([[0.980, 0.010, 0.010],
                            [0.015, 0.970, 0.015],
                            [0.020, 0.020, 0.960]])

    mu2 =  np.array([0.05, 0.95])
    s2 =  np.array([0.1, 0.1])
    pi2 = np.array([0.6, 0.3])

    transition2 = np.array([[0.98, 0.02],
                            [0.04, 0.96]])

    traces = []
    vits = []
    chains = []

    N1 = nmols//4
    T = nt
    K1 = np.arange(len(mu1), dtype = 'float64')

    dataset_number = 2

    seed = nrestarts*dataset_number+rep_number
    for j in range(N1):

        c =  generate_chain(T,K1,pi1, transition1, 666 + seed + j)
        i,t = add_emission(c,K1,mu1,s1)

        chains.append(c)
        traces.append(t)
        vits.append(i)

    K2 = np.arange(len(mu2), dtype = 'float64')

    for jj in range(nmols - N1):

        c =  generate_chain(T,K2,pi2, transition2, 666 + seed + N1 + jj)
        i,t = add_emission(c,K2,mu2,s2)

        chains.append(c)
        traces.append(t)
        vits.append(i)

    chains = np.array(chains)
    traces = np.array(traces)
    vits = np.array(vits)

    return traces, vits, chains

if __name__ == '__main__':

    traces,vits,chains= simulate_static3(1,1,200,1000)
    print (traces[20][0:10])
    plt.plot(traces[20])
    plt.show()

    plt.plot(vits[20], 'k')
    plt.show()

    print (traces[20][0:10])
    plt.plot(traces[-20])
    plt.show()

    plt.plot(vits[-20], 'k')
    plt.show()

    plt.hist(np.concatenate(traces),bins = 100)
    plt.show()
