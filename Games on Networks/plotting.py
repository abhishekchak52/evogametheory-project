import numpy as np 
import matplotlib.pyplot as plt
import glob
import os

files = [file[:-4] for file in glob.glob("*.npy")]

degree_list = []
for file in files:
    data = file.split('_')
    graph_size = int(data[0][1:])
    num_trials = int(data[1][1:])
    graph_degree = int(data[2][1:])
    degree_list.append(graph_degree)
    b_list, rho_list =  np.load(f'g{graph_size}_t{num_trials}_k{graph_degree}.npy')
    plt.plot(b_list, rho_list,'.')

neutral_fix_p = np.ones(100)/graph_size

plt.legend(degree_list)

plt.plot(np.linspace(0,20,100), neutral_fix_p, ls='--')
plt.xlim([0,20])
plt.show()