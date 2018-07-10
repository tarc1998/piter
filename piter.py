import math
import numpy as np

n = 5
m = 10
iter = 5
goal = (0, 0)
gamma = 1

probab = [0.6, 0.1, 0.1, 0.1]
dir_r = [-1, 0, 1, 0]
dir_c = [0, 1, 0, -1]
v1 = np.array([[0]*m]*n)
v2 = np.random.rand(n, m)
v3 = np.random.rand(n, m)

# reward=np.array([[[-1]*4]*m]*n)
reward = -np.ones((n,m,4))
prob = np.array([[[[[0.]*m]*n]*m]*n]*4)
pi = np.array([[[0.25]*4]*m]*n)
pi_prime = np.array([[[0.25]*4]*m]*n)

# setting dependent probabilities
for i in range(0, n):
    for j in range(0, m):
        for k in range(0, 4):
            stay = 1 - sum(probab)
            for l in range(0, 4):
                if 0 <= i+dir_r[(k+l) % 4]<n and 0 <= j+dir_c[(k+l) % 4]<m:
                    prob[k][i][j][i+dir_r[(k+l) % 4]][j+dir_c[(k+l) % 4]]=probab[l]
                else:
                    stay += probab[l]
            prob[k][i][j][i][j] = stay

# the upper left corner keeps
for a in range(0, 4):
    reward[0][0][a] = 0
    for i in range(0, n):
        for j in range(0, m):
            prob[a][0][0][i][j] = 0
    prob[a][0][0][0][0] = 1

# print(prob)


def policy_evaluation(pi, iter):
    v = np.array([[0] * m] * n)
    v_prime = np.array([[0] * m] * n)

    for it in range(iter):
        for i in range(n):
            for j in range(m):
                suma1 = 0
                for a in range(4):
                    suma2 = 0
                    for r in range(n):
                        for c in range(m):
                            suma2 += prob[a][i][j][r][c]*v[r][c]
                    suma1 += pi[i][j][a]*(reward[i][j][a]+gamma*suma2)
                v_prime[i][j]=suma1
        v = v_prime.copy()
    return v


def action_value(v):
    q = np.ones((n,m,4))
    for i in range(n):
        for j in range(m):
            for a in range(4):
                suma = 0
                for r in range(n):
                    for c in range(m):
                        suma += prob[a][i][j][r][c]*v[r][c]
                q[i][j][a]=reward[i][j][a]+gamma*suma
    return q


#print(action_value(policy_evaluation(pi, 100)))


def pimproved(q):
    pi = np.array([[[0.25] * 4] * m] * n)
    for i in range(n):
       for j in range(m):
           ar = np.argmax(q[i][j])
           for a in range(4):
               if a == ar:
                   pi[i][j][a]=1
               else:
                   pi[i][j][a]=0
    return pi

for i in range(iter):
    pi_prime = pimproved(action_value(policy_evaluation(pi, 100)))
    pi = pi_prime.copy()

disp = ["U", "R", "D", "L"]

for i in range(n):
    for j in range(m):
        print disp[np.argmax(pi[i, j])],
    print("")