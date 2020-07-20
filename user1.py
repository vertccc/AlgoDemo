import numpy as np
import random

# 参数维度
N = 10
# 迭代次数
H = 5
# 粒子个数
P = 100
# 位置边界
X_min = np.array([-30]*N)
X_max = np.array([30]*N)

R_rate = 0.7
M_rate = 0.8


def fitness(x):
    x = np.array(x)
    return -np.sum(np.power(x,2))

def random_x():
    return [random.choice(np.linspace(-30,30,100)) for x in range(N)]

# 初始化
population = [random_x() for _ in range(P)]


h = 0
while h < H:
    scores = [fitness(x) for x in population]
    print('{}, best: {:.2f}, ave: {:.2f}'.format(h,max(scores),np.mean(scores)))
    # 产生新的 X
    newx = []
    for i in range(P):
        if np.random.rand() < R_rate:
            # quan xin
            newx.append(random_x())
        else:
            new = random.choices(population,weights=scores)[0]
            # 变异
            for j in range(N):
                if np.random.rand() < M_rate:
                    new[j] = new[j] * (1+np.random.randn())
                    if new[j] > X_max[j]:
                        new[j] = X_max[j]
                    elif new[j] < X_min[j]:
                        new[j] = X_min[j]
            newx.append(new)
    # 计算新的 X的得分
    newscores = [fitness(x) for x in newx]
    # top P 个新旧X
    allpp = list(zip(population+newx,scores+newscores))
    allpp.sort(key=lambda x:x[1],reverse=True)
    allpp = allpp[:P]
    population = [x[0] for x in allpp]
    h += 1
   