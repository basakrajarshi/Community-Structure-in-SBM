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
import math

# Count time
start_time = time.time()

# Initialize all variables
e = 0
q = 2
n = int(1000)
k = int(n/q)
c = int(8)
p_in = (2*c + e)/(2*n)
p_out = (2*c - e)/(2*n)

# Initialize arrays for storing epidemic parameters, probabilities
# and number of runs for each probability
probabilities = np.arange(0.0, 1.001, 0.01)
epidemic_size = []
epidemic_length = []
runs = np.arange(1.0, 100.1, 1.0)


for pr in probabilities:

    total_infected_size = 0
    total_infected_length = 0

    for run in runs:

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
        
        total_infected_size += len(infected)
        total_infected_length += count

    # Find number of infected nodes and update array
    epidemic_size.append(total_infected_size/len(runs))
    # Find number of steps and update array
    epidemic_length.append(total_infected_length/len(runs))
    


# Modifications for plotting
epidemic_size_normalized = []

for i in epidemic_size:
    epidemic_size_normalized.append(i/(n))
    
for j in range(len(epidemic_size_normalized)):
    if (epidemic_size_normalized[j] >= 0.02):
        #print(probabilities[j])
        p_crit_1 = probabilities[j]
        break
    
for j in range(len(epidemic_size_normalized)):
    if (epidemic_size_normalized[j] >= 0.99):
        #print(probabilities[j])
        p_crit_2 = probabilities[j]
        break
    
for j in range(len(epidemic_length)):
    if (epidemic_length[j] >= 1.3):
        p_crit_3 = probabilities[j]
        break
    

    
#Plotting <s> vs. p
plt.scatter(probabilities,epidemic_size_normalized, alpha=0.5, color = 'b')
plt.xlabel('Probabilities p')
plt.ylabel('Average Epidemic size <s>')
#plt.ylim((0.9,3.1))
yy_1 = np.linspace(0.0, 1.0, num=100)
xx_1 = []
for i in yy_1:
    xx_1.append(p_crit_1)
plt.plot(xx_1,yy_1,linestyle='-', color='k', linewidth=1)
plt.savefig('s_versus_p_p-100_runs-100_3', dpi = 300)
plt.show()   

#Plotting <l> vs. p
plt.scatter(probabilities,epidemic_length, alpha=0.5, color = 'b')
plt.xlabel('Probabilities p')
plt.ylabel('Average Epidemic length <l>')
#plt.ylim((0.9,3.1))
yy_2 = np.linspace(1.0, max(epidemic_length), num=100)
xx_2 = []
for i in yy_2:
    xx_2.append(p_crit_3)
plt.plot(xx_2,yy_2,linestyle='-', color='k', linewidth=1)
xx_3 = np.linspace(0.0, 1.0, num=100)
yy_3 = []
for i in xx_3:
    yy_3.append(math.log(n))
plt.plot(xx_3,yy_3,linestyle='-.', color='k', linewidth=1)
plt.savefig('l_versus_p_p-100_runs-100_3', dpi = 300)
plt.show() 

# Record elapsed time
elapsed_time = time.time() - start_time

print(elapsed_time)