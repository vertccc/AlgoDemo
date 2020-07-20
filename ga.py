import random
import numpy as np

# 参数维度
N = 10
# 迭代次数
H = 5
# population
P = 100

M_rate = 0.3
C_rate = 0.6
X_min = np.array([-30]*N)
X_max = np.array([30]*N)

def random_chromosome():
    return [random.choice(np.linspace(-30,30,100)) for i in range(N)]
    
def fitness(chromosome):
    x = np.array(chromosome)
    return -np.sum(np.power(x,2))
    
def mutation(chromosome):
    for i in range(len(chromosome)):
        if np.random.rand() < M_rate:
            # do mutation
            new = chromosome[i]*(1+np.random.randn())
            if new > X_max[i]:
                new = X_max[i]
            elif new < X_min[i]:
                new = X_min[i]
            chromosome[i] = new
    return chromosome
    
def cross(pair):
    # 只能对应位置交换
    p1,p2 = pair
    for i in range(len(p1)):
        if np.random.rand() < C_rate:
            p1[i],p2[i] = p2[i],p1[i]
    return [p1,p2]

def div_pair(pop):
    random.shuffle(pop)
    pairs = []
    for i in range(0,len(pop),2):
        ele = (pop[i],pop[i+1])
        pairs.append(ele)
    return pairs


population = [random_chromosome() for i in range(P)]
h = 0
while h < H:
    # 计算适应度
    scores = [fitness(x) for x in population]
    ave = np.mean(scores)
    best = max(scores)
    bestx = population[scores.index(best)]
    # 选择,轮盘赌
    pop1 = random.choices(population,weights=scores,k=P)
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
    print('{}, score: {:.2f}, ave: {:.2f}'.format(h,best,ave))
    h += 1

    