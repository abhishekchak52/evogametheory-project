from multiprocessing import Pool
from numpy.random import randint, random


N=100   
num_trials = 10000

fitness_A = 0.99    # 1
fitness_B = 1 # 0

norm_fit_A = fitness_A/(fitness_A+fitness_B)
norm_fit_B = fitness_B/(fitness_A+fitness_B)

def fitness(member):
    return norm_fit_A if member else norm_fit_B


def constant_selection_moran_process(population):
    while True:
        if sum(population) in {0,N}:
            break
        else:
            while True:
                reproduce,replace = randint(0,N,size=2)
                if random() < fitness(population[reproduce]):
                    break
            population[replace] = population[reproduce]
    return sum(population)

species = [ [ 1 if i< 1 else 0 for i in range(N)] for j in range(4) ]

if __name__=='__main__':
    p = Pool(4)
    results = []
    for i in range(num_trials//4):
        set_result = p.map(constant_selection_moran_process,species)
        results+=set_result
        print('{} --> {}'.format(i*4,set_result))
    print("Fixation probability of type 1: {}".format(results.count(N)/num_trials))
    
    
