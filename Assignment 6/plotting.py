import numpy as np 
import matplotlib.pyplot as plt

if __name__ == '__main__':
    num_trials = 500000
    w_list = [0.01, 0.1, 1]
    fig, ax = plt.subplots(1,len(w_list), sharex= True, sharey= True, figsize=(len(w_list)*7,4))
    plt.suptitle(f'Stochastic simulation for Assignment 6 ({num_trials} trials )')
    for index, w in enumerate(w_list):
        N_sim, rho_sim = np.load(f'{num_trials}_nt/data_w_{w}.npy').T
        Nrho_sim = N_sim*rho_sim
        N_th, rho_th, Nrho_th = np.loadtxt(f'theoretical_w_{w}.txt',unpack=True)
        ax[index].set_title(f'w = {w}')
        ax[index].plot(N_th, Nrho_th, label='Theoretical', ls='--', color='orange' )
        ax[index].plot(N_sim, Nrho_sim,'b.', label = 'Simulation')
        ax[index].set_xlabel('Population size' if index == 1 else '')
        ax[index].set_ylabel('Rate of evolution'if index == 0 else '')
        ax[index].legend()
        
    filename = f'{num_trials}_nt/output.png'
    plt.savefig(filename)
    print(f'Saved to {filename}')
