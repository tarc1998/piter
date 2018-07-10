import math
import numpy as np

n = 5
m = 5
iter = 100
goal = (0, 0)
gamma = 1

dir_r = [-1, 0, 1, 0]
dir_c = [0, 1, 0, -1]
v1 = np.array([[0]*m]*n)
v2 = np.random.rand(n, m)
v3 = np.random.rand(n, m)

# reward=np.array([[[-1]*4]*m]*n)
reward = -np.ones((n,m,4))
prob = np.array([[[[[0]*m]*n]*m]*n]*4)
pi = np.array([[[0.25]*m]*n]*4)

# setting dependent probabilities
for i in range(0, n):
    for j in range(0, m):
        for k in range(0, 4):
            prob[k][i][j][i+dir_r[k]][j+dir_c[k]] = 0.6
            prob[k][i][j][i][j] = 0.1
            prob[k][i][j][i + dir_r[(k + 1) % 4]][j + dir_c[(k + 1) % 4]] = 0.1
            prob[k][i][j][i + dir_r[(k + 2) % 4]][j + dir_c[(k + 2) % 4]] = 0.1
            prob[k][i][j][i + dir_r[(k + 3) % 4]][j + dir_c[(k + 3) % 4]] = 0.1



# the upper left corner keeps
for dir in range(0, 4):
    reward[0][0][dir] = 0
    for i in range(0, n):
        for j in range(0, m):
            prob[dir][0][0][i][j] = 0
    prob[dir][0][0][0][0] = 1

u=1
for u in range(0,iter):
    v3=v1
    u=u+1
    if(u%1000 == 0): print(u)
    for i in range(0,n):
        for j in range(0,m):
            maks=-1000000
            for a in range(0,4):
                # v=reward[i][j][a]+gamma*sum(prob[a][i][j][r][c]*v1[r][c] for c in range(0,m) for r in range(0,n))
                suma=0
                for r in range(0,n):
                    for c in range(0,m):
                        suma+=prob[a,i,j,r,c]*v1[r,c]
                v=reward[i][j][a]+gamma*suma
                if v>maks:
                    maks=v
            v2[i][j]=maks
    v1=v2
print(u)
print(v1)
print(reward)