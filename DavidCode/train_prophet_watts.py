#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:42:20 2023

Train prophet model to predict final sizes on Erdos-Renyi random graphs
Then predict final sizes on a different test set of networks

@author: David
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import networkx as nx
from EpiNetSim import GillespieSim
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

def plot_training(values,label,fig_name):
    
    def moving_average(y,window=10):
        
        avg_y = []
        for idx in range(len(y) - window + 1):
            avg_y.append(np.mean(y[idx:idx+window]))
        for idx in range(window - 1):
            avg_y.insert(0, np.nan)
        
        return avg_y
    
    sns.set(style="darkgrid")
    fig, axs = plt.subplots(1, 1)
    sns.lineplot(x=list(range(len(values))), y=values, ax=axs)
    move_avg = moving_average(values)
    plt.plot(list(range(len(values))), move_avg, '--')
    axs.set_xlabel('Episode')
    axs.set_ylabel(label)
    fig.tight_layout()
    fig.set_size_inches(6, 4)
    fig.savefig(fig_name, dpi=200)

def generate(sim,p_vals):

    final_sizes = []
    nets = []
    for p in p_vals:
        rp = p
        G = nx.connected_watts_strogatz_graph(n, k, rp) # Generate a random Watts-Strogatz graph with contact prob p
        adj = nx.to_numpy_array(G) # adjacency matrix
        sim.beta = adj
        complete = False
        while not complete: # make sure sims completes to final time
            complete, f_size, times, I_traj = sim.run()
        final_sizes.append(f_size / n) # normalized by pop size
        nets.append(adj[:,:,np.newaxis])
        
    return np.array(final_sizes), np.array(nets)

# Sim params
n = 100 # pop size
k = 2 # nearest neighbors 
final_time = 10.0 # time simulations should end
init_I = np.zeros(n)
init_I[0] = 1 # seed first infection
rewire_p = .05  # rewiring probability
omega = 0 # immune waining: set to zero for SIR / Inf for SIS
nu = 0.5 # recovery/removal rate

""" 
    Init stochastic simulation
    Variables not supplied as keyword args will default to their defined values in __init__
"""    
sim = GillespieSim(pops=n,
                    terminate_condition="t[i] > 10",
                    final_time=final_time,
                    init_I=init_I,
                    beta=None,
                    omega=omega, 
                    d=nu)

"""
    Set up prophet model
    For Conv layers: first argument is the # of filters, second argument is the kernel size
    To think about: Should we sort rows/columns in adjacency matrix so most well-connected nodes are closer
    Maybe by single-linkage clustering: https://www.section.io/engineering-education/hierarchical-clustering-in-python/
"""
batch_size = 10 # num sims per training episode
input_shape = (batch_size,n,n,1)
prophet = keras.models.Sequential([
    keras.layers.Conv2D(2,8,activation="relu",padding="same",input_shape=input_shape[1:]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(4,4,activation="relu",padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(8,2,activation="relu",padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
    ])
prophet.summary()


"""
    Set training params
"""
#replay_buffer = deque(maxlen=2000)
optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.mean_squared_error

"""
    Generate training set of nets and final sizes
"""
p_vals = np.tile(np.linspace(0.0, 1.0, num=6),20)
train_final_sizes, train_nets = generate(sim, p_vals)

"""
    Train prophet on simulated nets/epidemics
"""
episode_losses = []
for episode in range(500):
    
    # Sample realizations for training batch
    batch_indices = np.random.choice(len(train_final_sizes), batch_size)
    net_batch = tf.convert_to_tensor(train_nets[batch_indices])
    true_final_sizes = tf.convert_to_tensor(train_final_sizes[batch_indices])
    true_final_sizes = tf.reshape(true_final_sizes, [batch_size,1])
    
    with tf.GradientTape() as tape:
       predicted_final_sizes = prophet(net_batch)
       loss = tf.reduce_mean(loss_fn(true_final_sizes,predicted_final_sizes))
    grads = tape.gradient(loss, prophet.trainable_variables)
    optimizer.apply_gradients(zip(grads, prophet.trainable_variables))   

    #if episode % 10 == 0:
    print('Episode: ' + str(episode) + '; Loss: ' + f'{loss.numpy():.3f}')

    episode_losses.append(loss.numpy())
    
plot_training(episode_losses,'Loss','loss_by_episode_watts.png')

"""
    For validation/test set: Simulate new batches of networks under different rewiring probs
    Compare error in predictions at each value with std error in epi sizes
"""
p_vals = np.tile(np.linspace(0.0, 1.0, num=6),100)
test_final_sizes, test_nets = generate(sim, p_vals)
predicted_final_sizes = prophet.predict(test_nets)
predicted_final_sizes = np.squeeze(predicted_final_sizes)
test_dict = {'Test Final Size': test_final_sizes, 'Predicted Final Size': predicted_final_sizes, 'Rewiring Prob': p_vals} 
df = pd.DataFrame(test_dict)
results_file = 'prophet_finalsize_predictions_watts.csv'
df.to_csv(results_file,index=False) 
    