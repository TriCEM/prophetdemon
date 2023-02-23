#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Train demon model to generate contact networks using a variational autoencoder (VAE)

For now just testing to see if the demon VAE can generate networks similar to random graphs

@author: David
"""

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np
import networkx as nx
from utils import plot_training

class CodingSampler(keras.layers.Layer):
    
    def call(self,inputs):
        mean, log_var = inputs
        codings = K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean
        return codings

class latent_loss(keras.losses.Loss):
        
    def call(self, codings_mean, codings_log_var):
        loss = -0.5 * K.sum(1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean),axis=-1)
        return loss


"""
    Generate training set of nets
"""
train_size = 100 
n = 100 # num nodes in contact network
p_vals = np.array([0.03]*train_size)
train_nets = []
for edge_p in p_vals:
    G = nx.gnp_random_graph(n, edge_p) # Generate a random Erdos-Renyi graph with contact prob p
    adj = nx.to_numpy_array(G) # adjacency matrix
    train_nets.append(adj[:,:,np.newaxis])
train_nets = np.array(train_nets)
print()

"""
    Set up the demon encoder model
"""
coding_dim = 10
inputs = keras.layers.Input(shape=[n,n])
z = keras.layers.Flatten()(inputs)
z = keras.layers.Dense(150,activation="selu")(z)
z = keras.layers.Dense(100,activation="selu")(z)
codings_mean = keras.layers.Dense(coding_dim)(z) # mean coding
codings_log_var = keras.layers.Dense(coding_dim)(z) # log var coding
codings = CodingSampler()([codings_mean,codings_log_var])
demon_encoder = keras.Model(inputs=[inputs], outputs=[codings_mean, codings_log_var, codings])
demon_encoder.summary()

"""
    Set up the demon decoder model
    Demon outputs an n x n set of probabilities that each pair is connected  
"""
decoder_inputs = keras.layers.Input(shape=[coding_dim])
x = keras.layers.Dense(100,activation="selu")(decoder_inputs)
x = keras.layers.Dense(150,activation="selu")(x)
x = keras.layers.Dense(n * n,activation="sigmoid")(x)
outputs = keras.layers.Reshape([n,n])(x)
demon_decoder = keras.Model(inputs=[decoder_inputs],outputs=[outputs])
demon_decoder.summary()

_, _, codings = demon_encoder(inputs)
reconstructions = demon_decoder(codings)
demon = keras.Model(inputs=[inputs], outputs=[codings_mean, codings_log_var, reconstructions])
demon.summary()

"""
    Set training params
"""
batch_size = 1
epochs = 1000
latent_loss_fn = latent_loss()
reconstruction_loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
optimizer = keras.optimizers.Adam(lr=1e-3)

"""
    Train demon to reconstruct ER-generated nets
    
    Note: Binary cross entropy computes mean over all elements instead of sum
    But latent loss sums over all latent variables
    We may therefore be underweighting reconstruction loss relative to latent loss
    See example here for computing losses on sums instead:
        https://keras.io/api/losses/probabilistic_losses/
    Reminder: We would want to take the mean loss over all instances in the batch if batch_size > 1
"""
episode_losses = []
for episode in range(epochs):
    
    # Sample realizations for training batch
    #batch_indices = np.random.choice(train_size, batch_size)
    #net_batch = tf.convert_to_tensor(train_nets[batch_indices])
    
    # For now just sample 1 training network at a time
    batch_indices = np.random.choice(train_size, batch_size)
    net_batch = tf.convert_to_tensor(train_nets[batch_indices])
    
    with tf.GradientTape() as tape:
       codings_mean, codings_log_var, reconstructions = demon(net_batch)
       lat_loss = latent_loss_fn(codings_mean, codings_log_var)
       rec_loss = reconstruction_loss_fn(net_batch,reconstructions)
       loss = lat_loss + rec_loss # latent loss plus reconstruction loss

    grads = tape.gradient(loss, demon.trainable_variables)
    optimizer.apply_gradients(zip(grads, demon.trainable_variables))   

    if episode % 10 == 0:
        print('Episode: ' + str(episode) + '; Loss: ' + f'{loss.numpy():.3f}' + '; Latent Loss: ' + f'{lat_loss.numpy():.3f}' + '; Reconstruction Loss: ' + f'{rec_loss.numpy():.3f}')

    episode_losses.append(loss.numpy())

plot_training(episode_losses,'Loss','loss_by_episode.png')

"""
    Compare statistical properties of demon-generated vs. ER-generated nets
"""
test_size = 100
rand_codings = tf.random.normal(shape=[test_size,coding_dim])
demon_probs = demon_decoder(rand_codings).numpy()
diffs = demon_probs - np.random.uniform(low=0.0, high=1.0, size=(test_size,n,n))
demon_test_nets = np.where(diffs < 0, 0, 1)
    

p_vals = np.array([0.03]*test_size)
ER_test_nets = []
for edge_p in p_vals:
    G = nx.gnp_random_graph(n, edge_p)
    ER_test_nets.append(G)

""" 
    Init stochastic simulation
    Variables not supplied as keyword args will default to their defined values in __init__
"""
from EpiNetSim import GillespieSim
final_time = 10.0 # time simulations should end
init_I = np.zeros(n)
init_I[0] = 1 # seed first infection
omega = 0 # immune waining: set to zero for SIR / Inf for SIS
nu = 0.5 # recovery/removal rate   
sim = GillespieSim(pops=n,
                    terminate_condition="t[i] > 10",
                    final_time=final_time,
                    init_I=init_I,
                    beta=None,
                    omega=omega, 
                    d=nu)

"""
    Compare statistical properties (connectivity,degree distribution,clustering)
    For more ideas see: https://networkx.org/documentation/stable/reference/algorithms/index.html
"""
mean_degree = [] # mean of degree dist
var_degree = [] # variance of degree dist
num_edges = [] # number of edges
clustering = [] # average clustering coefficient
components = [] # number connected components
diameter = [] # diameter - length of shortest path between most distant nodes
final_size = []
net_type = [] # type of network
for demon_net, ER_net in zip(demon_test_nets, ER_test_nets):
    
    demon_adj = np.triu(demon_net) + np.triu(demon_net).T # enforce symmetric adj requirment
    demon_net = nx.from_numpy_matrix(demon_adj)
    
    # Add stats for demon generated net
    mean_degree.append(np.mean(nx.degree_histogram(demon_net)))
    var_degree.append(np.var(nx.degree_histogram(demon_net)))
    num_edges.append(demon_net.number_of_edges())
    clustering.append(nx.average_clustering(demon_net))
    components.append(nx.number_connected_components(demon_net))
    if nx.is_connected(demon_net):
        diameter.append(nx.diameter(demon_net))
    else:
        diameter.append(np.inf)
    sim.beta = demon_adj
    complete = False
    while not complete: # make sure sims completes to final time
        complete, f_size, times, I_traj = sim.run()
    final_size.append(f_size / n) # normalized by pop size
    net_type.append('Demon')
    
    # Add stats for ER generated net
    mean_degree.append(np.mean(nx.degree_histogram(ER_net)))
    var_degree.append(np.var(nx.degree_histogram(ER_net)))
    num_edges.append(ER_net.number_of_edges())
    clustering.append(nx.average_clustering(ER_net))
    components.append(nx.number_connected_components(ER_net))
    if nx.is_connected(ER_net):
        diameter.append(nx.diameter(ER_net))
    else:
        diameter.append(np.inf)
    sim.beta = nx.to_numpy_array(ER_net)
    complete = False
    while not complete: # make sure sims completes to final time
        complete, f_size, times, I_traj = sim.run()
    final_size.append(f_size / n) # normalized by pop size
    net_type.append('ER')

    #print('Demon edges:' + str(demon_edges) + ' ;ER edges: ' + str(ER_edges))

import pandas as pd
test_dict = {'Mean Degree': mean_degree,
             'Var Degree': var_degree,
             'Num Edges': num_edges,
             'Clustering': clustering,
             'Components': components,
             'Diameter': diameter,
             'Final size': final_size,
             'Net Type': net_type} 
df = pd.DataFrame(test_dict)
results_file = 'demon_net_reconstruction_stats.csv'
df.to_csv(results_file,index=False) 
