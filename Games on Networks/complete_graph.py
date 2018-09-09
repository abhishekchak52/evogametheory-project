import numpy as np 
import matplotlib.pyplot as plt
import networkx as nx
from networkx.generators.classic import complete_graph
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired


def get_neighbours_list(G,node):
    adj = [(n,nbrdict) for n, nbrdict in G.adjacency() if n==node]
    return list(adj[0][1].keys())

def death_birth_fitness(G,node):
    b = 10
    c = 1
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
        fitness = num_C*b - c*(num_C+num_D)
        if G.nodes[nbr]['name'] == 'C':
                C_fitness += fitness
        elif G.nodes[nbr]['name'] == 'D':
                D_fitness += fitness
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
def evolve(graph):
    # hist = []
    while count_frequencies(graph)[0] not in [0.0,1.0]:
        chosen = np.random.randint(0,100)
        fn_C = death_birth_fitness(graph,chosen)[0]    
        # hist.append(count_frequencies(graph)[0])
        graph.nodes[chosen]['name'] = 'C' if np.random.random() < fn_C else 'D'
    return count_frequencies(graph)[0]
        # plt.show()
    # hist.append(count_frequencies(graph)[0])
    # plt.plot(hist)
    # plt.show()


if __name__=='__main__':
    num_trials = 100
    graph_size = 100
    num_workers = cpu_count()
    pool = ProcessPool()
    results = []
    invader = np.random.randint(graph_size)
    graphs = [complete_graph(graph_size)for _ in range(num_trials)]
    print('Generated graphs')
    # Initializing graph nodes
    for G in graphs:
        for node in G:
            G.nodes[node]['name'] = 'C' if node == invader else 'D'
    print('Initialised graphs')
    future = pool.map(evolve ,graphs,timeout=100)

    iterator = future.result()
    set_result = []
    # with tqdm(total=num_trials,desc='Finished',leave=False) as pbar:
    while True:
        try:
            result = next(iterator)
            print(result)
            results.append(result)
            
        except StopIteration:
            break
        except TimeoutError as error:
            # print("function took longer than %d seconds" % error.args[1])
            pass
        except ProcessExpired as error:
            # print("%s. Exit code: %d" % (error, error.exitcode))
            pass
        except Exception as error:
            pass
        # pbar.update(1)
                # print("function raised %s" % error)
                # print(error.traceback)  # Python's traceback of remote process

        # results+=set_result
    print(f'Fixation probability of type 1: {results.count(graph_size)/num_trials}')
    # invader = np.random.randint(graph_size)
    # graphs = [complete_graph(graph_size)for _ in range(num_workers)]
    # # Initializing graph nodes
    # for G in graphs:
    #     for node in G:
    #         G.nodes[node]['name'] = 'C' if node == invader else 'D'
    # evolve(G)