import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from generate_trace import generate_chain, add_emission

def simulate_reg(rep_number,nrestarts,nmols,nt):
    mu =  np.array([0.05, 0.95])
    s =  np.array([0.1, 0.1])
    pi = np.array([0.6, 0.3])
    transition = np.array([[0.98, 0.02],
                           [0.04, 0.96]])

    traces = []
    vits = []

    N = nmols
    T = nt
    K = np.arange(len(mu), dtype = 'float64')

    dataset_number = 0

    seed = nrestarts*dataset_number+rep_number
    for j in range(N):

        c =  generate_chain(T,K,pi, transition, 666 + seed + j)
        i,t = add_emission(c,K,mu,s)

        traces.append(t)
        vits.append(i)

    traces = np.array(traces)
    vits = np.array(vits)

    return traces, vits

if __name__ == '__main__':

    traces,vits = simulate_reg(1,1,200,1000)
    print (traces[20][0:10])
    plt.plot(traces[20])
    plt.plot(vits[20], 'k')
    plt.show()

    traces = np.array(traces)

    plt.hist(np.concatenate(traces),bins = 100)
    plt.show()
