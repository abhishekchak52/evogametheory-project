from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def fitness(member, population):
    b=10
    c=1
    w=0.01
    num_A = population.count('A')
    N = len(population)
    fitnesses = [
        (1-w+w*(b*num_A  - N*c)),
        (1-w+w*(b*num_A))
        ]
    # return fitnesses
    normed_fitness = [fitness/sum(fitnesses) for fitness in fitnesses]
    
    return normed_fitness[0] if member =='A' else normed_fitness[1]

def repeated_pd_game(population):
    N = len(population)
    # freq_history = []
    while True:
        # freq_history.append(population.count('A')/N)
        if population.count('A') in {0,N}: # Either species is fixed
            break
        else:            
            reproduce,replace = np.random.randint(0,N,size=2)
            if np.random.random() < fitness(population[reproduce], population):
                population[replace] = population[reproduce] 
    return population.count('A')/N
    # return freq_history

if __name__=='__main__':
    N=100
    num_trials = 100000
    population  =[[ 'A' if i< 1 else 'B' for i in range(N)] for _ in range(4)]
    pool = Pool(4)    
    results = []
    for i in trange(num_trials//4):
        set_result = pool.map(repeated_pd_game, population)
        results+=set_result
        # print('{} --> {}'.format(i*4,set_result))
    print(f'Fixation probability of type A: {results.count(N)/num_trials}')