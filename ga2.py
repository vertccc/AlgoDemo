import random
import numpy as np


CODE_L = 10
PROB_M = 0.1
PROB_C = 0.6


def random_chromosome():
    return [random.choice([0,1]) for i in range(CODE_L)]
    
def to_num(chromosome):
    chromosome = ''.join([str(x) for x in chromosome])
    x = 0 + int(chromosome,2) * (9/(2**CODE_L-1))
    return x

def fitness(chromosome):
    x = to_num(chromosome)
    return x + 10*np.sin(5*x) + 7*np.cos(4*x)
    
def mutation(chromosome):
    yes_m = np.random.rand(len(chromosome)) < PROB_M
    for i,m in enumerate(yes_m):
        if m:
            if chromosome[i] == 0:
                chromosome[i] = 1
            else:
                chromosome[i] = 0
    return chromosome
    
def cross(pair):
    if np.random.rand() > PROB_C:
        return [pair[0],pair[1]]
    length = 3
    p1,p2 = pair
    loc = np.random.randint(0,len(p1)-length-1)   
    r1 = p1.copy()
    r1[loc:loc+length] = p2[loc:loc+length].copy()
    r2 = p2.copy()
    r2[loc:loc+length] = p1[loc:loc+length].copy()
    return [r1,r2]

def div_pair(pop):
    random.shuffle(pop)
    pairs = []
    for i in range(0,len(pop),2):
        ele = (pop[i],pop[i+1])
        pairs.append(ele)
    return pairs

def genetic_algo(n_gen,pop_size):
    population = [random_chromosome() for i in range(pop_size)]
    n = 0
    while n < n_gen:
        # 计算适应度
        scores = [fitness(x) for x in population]
        # 选择,轮盘赌
        pop1 = random.choices(population,weights=scores,k=pop_size)
        # 交叉
        pairs = div_pair(pop1)
        pop2 = []
        for pair in pairs:
            new_pair = cross(pair)
            pop2.extend(new_pair)
        # 变异
        pop3 = [mutation(x) for x in pop2]
        # 迭代
        population = pop3.copy()
        n += 1
    return population


# ------------------------------------------------
if __name__ == '__main__':
    
   
    n_gen = 6
    pop_size = 100
    
    population = genetic_algo(n_gen,pop_size)

    
    scores = [fitness(x) for x in population]
    ans = population[scores.index(max(scores))]
    print('mean: {:.2f}, std: {:.2f}'.format(np.mean(scores),np.std(scores)))
    print('f(x) = {}, x = {}'.format(fitness(ans),to_num(ans)))

