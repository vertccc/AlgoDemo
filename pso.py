import numpy as np
import random


# 参数维度
N = 10
# 迭代次数
H = 10
# 粒子个数
P = 100
# 学习常数
WMAX = 1
WMIN = 0.5
C1,C2 = 0.6,3
# 速度边界
V_min = np.array([-20]*N)
V_max = np.array([20]*N)
# 位置边界
X_min = np.array([-30]*N)
X_max = np.array([30]*N)


def fitness(x):
    return -np.sum(np.power(x,2))

# 初始化
X = []
V = []
for i in range(P):
    p = [random.choice(np.linspace(-30,30,100)) for _ in range(N)]
    v = [random.choice(np.linspace(-20,20,100)) for _ in range(N)]
    X.append(np.array(p))
    V.append(np.array(v))

h = 0
pbest_s = [-np.inf]*P
pbest_p = [None]*P
gbest_s = -np.inf
gbest_p = None
while h < H:
    # 计算得分，更新best
    scores = []
    for i,x in enumerate(X):
        s = fitness(x)
        scores.append(s)
        if s > gbest_s:
            gbest_s = s
            gbest_p = x
        if s > pbest_s[i]:
            pbest_s[i] = s
            pbest_p[i] = x
    # 更新v
    w = WMAX - (WMAX-WMIN)*h/H
    for i,v in enumerate(V):
        newv = w * V[i] + C1*np.random.randn()*(pbest_p[i]-X[i]) \
                + C2*np.random.randn()*(gbest_p-X[i])
        # limit
        maxmask = newv>V_max
        minmask = newv<V_min
        newv[maxmask] = V_max[maxmask]
        newv[minmask] = V_min[minmask]
        V[i] = newv
        # update loaction
        newx = X[i] + newv
        maxmask = newx>X_max
        minmask = newx<X_min
        newx[maxmask] = X_max[maxmask]
        newx[minmask] = X_min[minmask]
        X[i] = newx
    print('{} score:{:.2f}, ave: {:.2f}'.format(h,gbest_s,np.mean(scores)))
    h += 1
        
            
            
        
        