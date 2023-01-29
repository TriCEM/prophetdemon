#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:15:41 2023

Test ER random graph generation in networkx

@author: David
"""

import matplotlib.pyplot as plt
import networkx as nx

n = 10  # 10 nodes
p = .25  # edge probability
seed = 20160  # seed random number generators for reproducibility

# Use seed for reproducibility
G = nx.gnp_random_graph(n, p, seed=seed)

# some properties
print("node degree clustering")
for v in nx.nodes(G):
    print(f"{v} {nx.degree(G, v)} {nx.clustering(G, v)}")

print()
print("the adjacency list")
for line in nx.generate_adjlist(G):
    print(line)

print()
print("number of nodes")
print(G.number_of_nodes())

print()
print("number of edges")
print(G.number_of_edges())

print()
print("get degree dist")
print(nx.degree_histogram(G))

print()
print("numpy array of adj matrix")
print(nx.to_numpy_array(G))



#sub = plt.subplot()
# pos = nx.spring_layout(G, seed=seed)  # Seed for reproducible layout
# nx.draw(G, pos=pos)
# plt.show()