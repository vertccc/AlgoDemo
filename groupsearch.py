import random
import numpy as np


def se_nse(se,n):
    s,e = se
    glen = (e-s+1)//n
    if glen == 0:
        return False,[se]
    gps = [(s+i*glen,s+(i+1)*glen-1) for i in range(n)]
    return True,gps

def ten_spec(n,jinwei):
    value = [0]*len(jinwei)
    i = -1
    while(True):
        yushu = n % (jinwei[i])
        n = n//jinwei[i]
        value[i] = yushu
        i -= 1
        if n == 0:
            break
    return value

def get_para(n,xlst):
    jinwei = [len(x) for x in xlst]
    para_idx = ten_spec(n,jinwei)
    para = [xlst[i][para_idx[i]] for i in range(len(para_idx))]
    return para

def fitness(x):
    return -np.sum(np.power(x,2))

xlst = [np.linspace(-30,30,100)] * 10

n,topm = 1000,500


j = 0
can_loop = True
jinwei = [len(x) for x  in xlst]
rr = 1
for x in jinwei:
    rr = rr*x
    total_range = (0,rr-1)
_,groups = se_nse(total_range,n)

while can_loop:
    if groups[0][0] == groups[0][1]:
        para_idxs = [s for s,e in groups]
    else:
        para_idxs = [s+int(random.uniform(0,1)*(e-s)) for s,e in groups]

    paralst = [get_para(x,xlst) for x in para_idxs]
    idx_scores = [(i,fitness(x)) for i,x in enumerate(paralst)]
    idx_scores.sort(key=lambda x:x[1],reverse=True)
    best = idx_scores[0][1]
    ave = np.mean([x[1] for x in idx_scores])
    print('{}, best: {:.2f}, ave: {:.2f}'.format(j,best,ave))
    # get topm
    top_scores = idx_scores[:topm]
    top_groups = [groups[x[0]] for x in top_scores]
    new_groups = []
    hh = n//topm
    for gp in top_groups:
        can_split,ngps = se_nse(gp,hh)
        if can_split:
            new_groups.extend(ngps)
    if len(new_groups) < n:
        can_loop = False
        groups = top_groups[0]
    else:
        can_loop = True
        groups = new_groups
    j += 1

ss = [fitness(get_para(x,xlst)) for x in groups]
print('{}, best: {:.2f}, ave: {:.2f}'.format(j,max(ss),np.mean(ss)))

