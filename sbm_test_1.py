# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 23:39:52 2018

@author: rajar
"""

import networkx as nx
import matplotlib.pyplot as plt

n = int(50)
c = int(5)
e = [0,4,8]
sizes = [25, 25]

for eps in e:
    p_in = (2*c + eps)/(2*n)
    p_out = (2*c - eps)/(2*n)
    probs = [[p_in, p_out],
             [p_out, p_in]]
    g = nx.stochastic_block_model(sizes, probs, seed=0)
    if (eps == 0):
        nx.draw(g)
        plt.savefig('graph_vis_eps_0', dpi = 300)
        plt.show()
    elif (eps == 4):
        nx.draw(g)
        plt.savefig('graph_vis_eps_4', dpi = 300)
        plt.show()
    elif (eps == 8):
        nx.draw(g)
        plt.savefig('graph_vis_eps_8', dpi = 300)
        plt.show()

