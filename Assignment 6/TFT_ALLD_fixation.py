from multiprocessing import Pool, cpu_count 
import numpy as np
from tqdm import tqdm
import os
from functools import partial

R = 3
S = 0
T = 5
P = 1

m = 10 # number of rounds
w_list = [0.01, 0.1, 1]
population_sizes = [100, 150, 200, 250, 300, 350, 400, 600, 800, 1600]


def fitness(member, population, w):
    i = population.count('A')
    N = len(population)
    Fi = (m*R*(i-1) + (S+(m-1)*P)*(N-i))/(N-1)
    Gi = ((T+(m-1)*P)*i + m*P*(N-i-1))/(N-1)
    fi = 1 - w + w*Fi
    gi = 1 - w + w*Gi

    fitnesses = [fi, gi]
    normed_fitness = [fitness/sum(fitnesses) for fitness in fitnesses]
    return normed_fitness[0] if member == 'A' else normed_fitness[1]

def TFT_ALLD_invasion(N, w):
    population = ['A' if i < 1 else 'B' for i in range(N)]
    while True:
        if population.count('A') in {0, N}: # Either species is fixed
            break
        else:
            reproduce, replace = np.random.randint(0, N, size=2)
            if np.random.random() < fitness(population[reproduce], population, w):
                population[replace] = population[reproduce]
    return population.count('A')/N

if __name__ == '__main__':
    num_trials = 100000
    num_workers = 250
    os.makedirs(f'{num_trials}_nt', exist_ok=True)
    os.chdir(f'{num_trials}_nt')
    for w in tqdm(w_list,desc="Varying Selection",leave=True):
        rho = []
        for size in tqdm(population_sizes,desc="Sizes",leave=False):
            with Pool(num_workers) as pool:
                results = []
                for _ in tqdm(range(num_trials//num_workers),leave=False,desc=f'Pop. size = {size}'):
                    sizes = [size] * num_workers
                    set_results = pool.map(partial(TFT_ALLD_invasion, w=w), sizes)
                    results += set_results
                rho.append(sum(results)/num_trials)
        np.save(f'data_w_{w}', np.array(list(zip(population_sizes,rho))))
