from multiprocessing import Pool
from numpy.random import randint, random


N=100   
num_trials = 1000

fitness_A = 0.99 # 0
fitness_B = 1    # 1

norm_fit_A = fitness_A/(fitness_A+fitness_B)
norm_fit_B = fitness_B/(fitness_A+fitness_B)

def fitness(member):
    return norm_fit_A if member=='A' else norm_fit_B


def disadvantageous_mutant_invasion(population):
    while True:
        if population.count('A') in {0,N}:
            break
        else:
            reproduce,replace = randint(0,N,size=2)
            if random() < fitness(population[reproduce]):
                population[replace] = population[reproduce]
    return population.count('A')/N

species = [ [ 'A' if i< N//2 else 'B' for i in range(N)] for j in range(4) ]

if __name__=='__main__':
    p = Pool(4)
    results = []
    for i in range(num_trials//4):
        set_result = p.map(disadvantageous_mutant_invasion,species)
        results+=set_result
        print(set_result)
    print("Fixation probability of type A: {}".format(results.count(1)/num_trials))