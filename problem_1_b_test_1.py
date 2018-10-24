# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 17:22:40 2018

@author: rajar
"""

import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

e = 0
q = 2
n = int(1000)
k = int(n/q)
c = int(8)
p_in = (2*c + e)/(2*n)
p_out = (2*c - e)/(2*n)

probabilities = np.arange(0.0, 1.001, 0.001)
epidemic_size = []
epidemic_length = []

for pr in probabilities:

    graph = nx.planted_partition_graph(q, k, p_in, p_out, seed=42
                                       , directed=False)
    #nx.draw(graph)
    # Append all nodes to the list nodes
    nodes = []
    for i in graph.nodes():
        nodes.append(i)
    
    #print(random.randint(0,n))
    # Initialize variables
    infected = []
    inf = random.randint(0,n-1)
    infected.append(inf)
    to_be_infected = [n-1]
    count = 0
    
    # while new_infected is not equal to infected
    # or no new nodes are infected
    while (to_be_infected != []):
        to_be_infected = []
        # Generate a random number between 0 and 1 
        #prob = random.uniform(0,1)
        # Check if the generated number is less than or equal to p
        #if (prob <= p_in and prob <= p_out):
            # Add all of the current node in 'infected' list's neighbours 
            # to 'infected' that are not in 'infected'
            #print(infected)
        for i in infected:
            #print('Already infected node:',i)
            for j in graph.neighbors(i):
                #print('Neighbors of infected node' ,i, 'is node' ,j)
                if (j not in infected):
                    prob = random.uniform(0,1)
                    if (prob <= pr):
                        to_be_infected.append(j)
                        #print('New nodes added')
                    infected = list(set(infected) | set(to_be_infected))
                    #print('The nodes to be infected are',to_be_infected)
                    #print('The     new infected nodes are' ,infected)
            #to_be_infected = []
        if (to_be_infected is not None):
            count += 1
    # Find number of infected nodes and update array
    epidemic_size.append(len(infected))
    # Find number of steps and update array
    epidemic_length.append(count)
    
#Plotting <s> vs. p
plt.scatter(probabilities,epidemic_size, s=2, alpha=0.5, color = 'b')
plt.xlabel('Probabilities p')
plt.ylabel('Epidemic size s')
#plt.ylim((0.9,3.1))
plt.savefig('s_versus_p_5', dpi = 300)
plt.show()   

#Plotting <l> vs. p
plt.scatter(probabilities,epidemic_length, s=2, alpha=0.5, color = 'b')
plt.xlabel('Probabilities p')
plt.ylabel('Epidemic length l')
#plt.ylim((0.9,3.1))
plt.savefig('l_versus_p_5', dpi = 300)
plt.show() 

elapsed_time = time.time() - start_time

print(elapsed_time)