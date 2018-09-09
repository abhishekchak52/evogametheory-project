from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

R = 3
S = 0
T = 5
P = 1

m = 10 # number of rounds
w = 0.01

population_sizes = [100, 150, 200, 250, 300, 350, 400, 600, 800, 1600]
num_trials = 1000



def fitness(member, population):
    
    i = population.count('A')
    N = len(population)
    Fi = (m*R*(i-1) + (S+(m-1)*P)*(N-i))/(N-1)
    Gi = ((T+(m-1)*P)*i + m*P*(N-i-1))/(N-1)
    fi = 1 - w + w*Fi
    gi = 1 - w + w*Gi

    fitnesses = [fi, gi]
    normed_fitness = [fitness/sum(fitnesses) for fitness in fitnesses]
    
    return normed_fitness[0] if member =='A' else normed_fitness[1]
    
def TFT_ALLD_invasion(population):
    N = len(population)
    while True:
        if population.count('A') in {0,N}: # Either species is fixed
            break
        else:            
            reproduce,replace = np.random.randint(0,N,size=2)
            if np.random.random() < fitness(population[reproduce], population):
                population[replace] = population[reproduce] 
    return population.count('A')/N


if __name__=='__main__':
    pool = Pool()
    results = []
    rho_tft = []
    Nrho_tft = []
    for size in population_sizes:
        for i in tqdm(range(num_trials//cpu_count()),desc=f'Size = {size}', leave=False ):
            species = [ [ 'A' if i<1 else 'B' for i in range(size)] for _ in range(cpu_count) ]
            set_result = pool.map(TFT_ALLD_invasion,species)
            results+=set_result
        rho_tft.append(sum(results)/num_trials)
        Nrho_tft.append(size*sum(results)/num_trials)
        print(f'Fixation probability of type A for population size {size}: {sum(results)/num_trials}')
    np.save(f'w_{w}',rho_tft)
    plt.plot(population_sizes, Nrho_tft,'.')
    plt.show()

    
    
