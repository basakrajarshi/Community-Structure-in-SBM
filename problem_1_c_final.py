# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 01:00:19 2018

@author: rajar
"""

import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import math
from mpl_toolkits.mplot3d import Axes3D

# Count time
start_time = time.time()

# Initialize all variables
# e = 0
q = 2
n = int(200)
k = int(n/q)
c = int(8)
eps = np.arange(0.0, 2*c + 0.01, 0.1)

array_epsilon = []
array_probability = []
array_epi_size = []
array_epi_len = []

for e in eps:

    # Define edge drawing probabilities
    p_in = (2*c + e)/(2*n)
    p_out = (2*c - e)/(2*n)

    # Initialize arrays for storing epidemic parameters, probabilities
    # and number of runs for each probability
    probabilities = np.arange(0.0, 1.001, 0.01)
    epidemic_size = []
    epidemic_length = []
    runs = np.arange(0.0, 1.01, 0.01)


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
                            
                if (to_be_infected is not None):
                    count += 1
            
            total_infected_size += len(infected)
            total_infected_length += count

        array_epsilon.append(e) # Array for the x-axis
        array_probability.append(pr) # Array for the y-axis
        array_epi_size.append(total_infected_size/len(runs)) # Array for the z(1) axis
        array_epi_len.append(total_infected_length/len(runs)) # Array for the z(2) axis

    
# Modifications for plotting
array_epi_size_norm = []

for i in array_epi_size:
     array_epi_size_norm.append(i/(n))
    

mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(array_epsilon, array_probability, array_epi_size_norm)
ax.set_xlabel('Epsilon e')
ax.set_ylabel('Probability p')
ax.set_zlabel('Average Epidemic Size <s>')
ax.legend()
plt.savefig('s_vs_p_vs_e_p-100_e-160_runs-100', dpi = 300)
plt.show()


mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(array_epsilon, array_probability, array_epi_len)
ax.set_xlabel('Epsilon e')
ax.set_ylabel('Probability p')
ax.set_zlabel('Average Epidemic Length <l>')
ax.legend()
plt.savefig('l_vs_p_vs_e_p-100_e-160_runs-100', dpi = 300)
plt.show()


mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(array_probability, array_epsilon, array_epi_size_norm)
ax.set_xlabel('Probability p')
ax.set_ylabel('Epsilon e')
ax.set_zlabel('Average Epidemic Size <s>')
ax.legend()
plt.savefig('s_vs_e_vs_p_p-100_e-160_runs-100', dpi = 300)
plt.show()


mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(array_probability, array_epsilon, array_epi_len)
ax.set_xlabel('Probability p')
ax.set_ylabel('Epsilon e')
ax.set_zlabel('Average Epidemic Length <l>')
ax.legend()
plt.savefig('l_vs_e_vs_p_p-100_e-160_runs-100', dpi = 300)
plt.show()


# Record elapsed time
elapsed_time = time.time() - start_time

print((elapsed_time/3600), 'hours' )