import os
from multiprocessing import Pool
import time
from functools import partial
import numpy as np 
import matplotlib.pyplot as plt
import networkx as nx
from networkx.generators.random_graphs import random_regular_graph
from tqdm import tqdm


graph_size = 100
num_trials = 100000
network_refresh_interval = 5000
num_refreshes = num_trials//network_refresh_interval
num_workers = 250
graph_degree = 8


def get_neighbours_list(G,node):
    adj = [(n,nbrdict) for n, nbrdict in G.adjacency() if n==node]
    return list(adj[0][1].keys())

def death_birth_fitness(G,node,b):
    c = 1
    w=0.01
    C_fitness = D_fitness = 0
    # get all neighbours of cell
    neighbours = get_neighbours_list(G,node)
    for nbr in neighbours:
        # find it's neighbours
        nbr_list = get_neighbours_list(G,nbr)
        num_C = num_D = 0
        for nb in nbr_list:   
            if G.nodes[nb]['name'] == 'C':
                num_C += 1
            elif G.nodes[nb]['name'] == 'D':
                num_D += 1
        if G.nodes[nbr]['name'] == 'C':
                C_fitness += 1-w+w*(num_C*b - c*(num_C+num_D))
        elif G.nodes[nbr]['name'] == 'D':
                D_fitness += 1-w+w*(num_C*b)
    fitnesses = [C_fitness, D_fitness]
    return np.array(fitnesses)/sum(fitnesses)

def count_frequencies(G):
    num_C = num_D = 0
    for node in G:
        if G.nodes[node]['name'] == 'C':
                num_C += 1
        elif G.nodes[node]['name'] == 'D':
                num_D += 1
    nums = [num_C, num_D]
    return np.array(nums)/sum(nums)

def list_players(G):
    players = []
    for node in G:
        players.append(G.nodes[node]['name'])
    return players

def evolve(graph,b):
    while count_frequencies(graph)[0] not in [0.0,1.0]:
        chosen = np.random.randint(0,graph_size)
        fn_C = death_birth_fitness(graph,chosen,b)[0]    
        graph.nodes[chosen]['name'] = 'C' if np.random.random() < fn_C else 'D'
    return count_frequencies(graph)[0]

if __name__=='__main__':
    
    os.makedirs(f'{num_trials}_nt', exist_ok=True)
    os.chdir(f'{num_trials}_nt')
    b_list = np.linspace(3,5,10)
    rho_list = []
    with Pool(num_workers) as pool:
        for benefit in tqdm(b_list, leave=False, desc='benefits'):
            results = []
            for i in tqdm(range(num_refreshes),leave=False,desc=f'benefit = {benefit}'):
                for j in tqdm(range(network_refresh_interval//num_workers),desc='Batch', leave=False):
                    # Generate a fresh graph with a random cooperator
                    invader = np.random.randint(graph_size)      
                    graphs = [random_regular_graph(graph_degree,graph_size,seed=time.time())for _ in range(num_workers)]
                    # Initializing graph nodes
                    for G in graphs:
                        for node in G:
                            G.nodes[node]['name'] = 'C' if node == invader else 'D'
                    set_result = pool.map(partial(evolve,b=benefit) ,graphs)
                    results+=set_result
                    # print('{} --> {}'.format(i*network_refresh_interval+j*num_workers,set_result))
            # print("\nFixation probability of C: {}".format(sum(results)/num_trials))
            rho_list.append(sum(results)/num_trials)
    plt.plot(b_list,rho_list,'o')
    neutral_fix_p = 0.01*np.ones_like(b_list)

    np.save(f'g{graph_size}_{graph_degree}',np.array([b_list,rho_list]))

    plt.plot(b_list,neutral_fix_p,ls='--')
    plt.savefig(f'g{graph_size}_k{graph_degree}.png')
    plt.show()    

