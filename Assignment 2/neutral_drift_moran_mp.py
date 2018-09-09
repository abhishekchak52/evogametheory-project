from multiprocessing import Pool
from numpy.random import randint
import matplotlib.pyplot as plt

N = 500
num_trials = 100


def neutral_drift_moran_process(population):
    while True:
        if sum(population) in {0, N}:
            break
        else:
            reproduce, replace = randint(0, N, size=2)
            population[replace] = population[reproduce]
    return sum(population)


species = [[0 if i < N / 2 else 1 for i in range(N)] for j in range(4)]

if __name__ == "__main__":
    p = Pool(4)
    results = []
    for i in range(num_trials // 4):
        set_result = p.map(neutral_drift_moran_process, species)
        results += set_result
        print(set_result)
    print("Fixation probability of type 1: {}".format(results.count(500) / num_trials))
