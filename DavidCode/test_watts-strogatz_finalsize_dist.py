#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:09:02 2023

Test implmentation of epi net sims on random graphs
Performs n stochastic sims and outputs final size distribution

@author: David
"""
import numpy as np
from EpiNetSim import GillespieSim
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Init configuration of infected individuals on network
n = 100 # pop size
k = 2 # nearest neighbors 
final_time = 10.0 # time simulations should end
init_I = np.zeros(n)
init_I[0] = 1 # seed first infection
rewire_p = .05  # rewiring probability
omega = 0 # immune waining: set to zero for SIR / Inf for SIS
nu = 0.5 # recovery/removal rate

# Set up net param to vary
rewiring_p_vals = np.tile(np.linspace(0.0, 1.0, num=6),20)
param_vals = np.unique(rewiring_p_vals)

# Run stochastic sims to compare exact probs to approx probs
n_sims = len(rewiring_p_vals)
final_sizes = []
time_trajectories = []
I_trajectories = []
for s in range(n_sims):
    
    print("Sim # :" + str(s))
    
    # Generate a random Erdos-Renyi graph with contact prob p
    rp = rewiring_p_vals[s]
    G = nx.connected_watts_strogatz_graph(n, k, rp)
    adj = nx.to_numpy_array(G) # adjacency matrix
    print("Average degree: ",2*G.number_of_edges()/G.number_of_nodes())

    # Init stochastic simulation -- variables not supplied as keyword args will default to their defined values in __init__
    sim = GillespieSim(pops=n,
                        terminate_condition="t[i] > 10",
                        final_time=final_time,
                        init_I=init_I,
                        beta=adj,
                        omega=omega, 
                        d=nu)
    
    complete = False
    while not complete: # make sure sims completes to final time
        complete, f_size, times, I_traj = sim.run()
    final_sizes.append(f_size)
    time_trajectories.append(times)
    I_trajectories.append(I_traj)

print("Final sizes: " + str(final_sizes))

"""
    Plot prevalence trajectories
"""
sns.set()
pal = sns.color_palette("magma", n_colors=6)
sns.set_palette(
    palette="magma",
    n_colors=6,
)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
points = 50 # desired number of time points on plot
cmap = {v:c for v,c in zip(param_vals,pal)}
for s in range(n_sims):

    t = time_trajectories[s]
    I = I_trajectories[s] 
    plot_freq = max(1,int(len(t) / points))
    I = I[::plot_freq]
    t = t[::plot_freq]
    ax.plot(t, I, color=cmap[rewiring_p_vals[s]],linewidth=2,alpha=.5) 
    
ax.set_xlabel('Time')
ax.set_ylabel('Prevalence')
fig.tight_layout()
png_file = 'sim_watts-strogatz_trajs.png'
fig.savefig(png_file, dpi=200)

"""
    Plot final size distribution at different param values
"""
test_dict = {'Final Size': final_sizes, 'Rewiring Prob': rewiring_p_vals} 
df = pd.DataFrame(test_dict)
#results_file = 'sim_ER_finalsizes.csv'
#df.to_csv(results_file,index=False)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
sns.violinplot(data=df, x="Rewiring Prob", y="Final Size",inner="points")
ax.set_xticklabels([f'{x:.2f}' for x in param_vals])

fig.tight_layout()
png_file = 'sim_watts-strogatz_finalsize_by_rewireprobs.png'
fig.savefig(png_file, dpi=200)     

