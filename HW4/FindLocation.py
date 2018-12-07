import numpy as np
import random

D = np.matrix([[0, 206, 429, 1504, 963, 2976, 3095, 2979, 1949],
	 [206, 0, 233, 1308, 802, 2815, 2934, 2786, 1771],
	 [429, 233, 0, 1075, 671, 2684, 2799, 2631, 1616],
	 [1504, 1308, 1075, 0, 1329, 3273, 3053, 2687, 2037],
	 [963, 802, 671, 1329, 0, 2013, 2142, 2054, 996],
	 [2976, 2815, 2684, 3273, 2013, 0, 808, 1131, 1307],
	 [3095, 2934, 2799, 3053, 2142, 808, 0, 379, 1235],
	 [2979, 2786, 2631, 2687, 2054, 1131, 379, 0, 1059],
	 [1949, 1771, 1616, 2037, 996, 1307, 1235, 1059, 0]])

alpha = 0.01
niter = 10000
L = {}
Cities = ['BOS', 'NYC', 'DC', 'MIA', 'CHI', 'SEA', 'SF', 'LA', 'DEN']
n = len(Cities)
for i in Cities:
	L[i] = np.array([random.randrange(0, 4000)+0.1, random.randrange(0, 4000)+0.1])

def SGD(L):
	while niter > 0:
		for i in range(n):
			der = np.zeros((2))
			for j in range(n):
				if j != i:
					s = L[i] - L[j]
                	der += (1 - D[i, j]/np.linalg.norm(s)) * 2 * s
                    #print(i, j, der)
        	L[i] -= alpha * der
		niter -= 1
	return L

def BGD(L):
	while niter > 0:
	    tmp = np.zeros((n, 2))
	    for i in range(n):
	        der = np.zeros(2)
	        for j in range(n):
	            if j != i:
	                s = L[i] - L[j]
	                der += (1 - D[i, j]/np.linalg.norm(s)) * 2 * s
	                    #print(i, j, der)
	        tmp[i] = der
	    L -= alpha * tmp
	    niter -= 1
	return L


import matplotlib.pyplot as plt
colors = ['b','g','r','c','m','y','k','grey','purple']
for i in range(n):
    plt.scatter(L[i,0], L[i,1], c= colors[i], label=Cities[i])#, color=)
plt.legend()
plt.show() 
