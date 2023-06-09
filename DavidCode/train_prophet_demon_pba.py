#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Train prophet-demon model asynchronouly:
    -First train prophet to accurately predict final sizes
    -Then train demon VAE to deteriorate the predictive perfomance of the prophet

For now just testing to see if the demon VAE can generate networks similar to random graphs

Note: this version uses the Poisson-Binomial appoximation to compute final size distributions

@author: David
"""

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np
import networkx as nx
from utils import plot_training
from EpiNetSim import PoissonBinomialModel

class CodingSampler(keras.layers.Layer):
    
    def call(self,inputs):
        mean, log_var = inputs
        codings = K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean
        return codings

class latent_loss(keras.losses.Loss):
        
    def call(self, codings_mean, codings_log_var):
        loss = -0.5 * K.sum(1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean),axis=-1)
        return loss

def build_prophet(n,batch_size,summary=True):
    
    """
        Build prophet model as a CNN conecteed to a FFNN
         n (int): network node size
         batch_size (int): number of nets to train on each training iteration
         summary (boolean): Print summary of keras model architecture
         
    """
    
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
    
    if summary:
        prophet.summary()
        
    return prophet

def train_prophet(prophet, train_final_sizes, train_nets, iterations=500):
    
    """
        Train prophet model on training set
    """
    #replay_buffer = deque(maxlen=2000)
    optimizer = keras.optimizers.Adam(lr=1e-3)
    loss_fn = keras.losses.mean_squared_error

    episode_losses = []
    for episode in range(iterations):
        
        # Sample realizations for training batch
        batch_indices = np.random.choice(len(train_final_sizes), batch_size)
        net_batch = tf.convert_to_tensor(train_nets[batch_indices])
        true_final_sizes = tf.convert_to_tensor(train_final_sizes[batch_indices])
        true_final_sizes = tf.reshape(true_final_sizes, [batch_size,1])
        
        with tf.GradientTape() as tape:
           predicted_final_sizes = prophet(net_batch)
           
           # If using Poisson-Binomial final size distribution
           #final_size_idx = (predicted_final_sizes.numpy()*n).flatten().round().astype(int)
           #indices = tf.constant([[x,y] for x,y in zip(range(10),final_size_idx)])
           #final_size_probs = tf.gather_nd(true_final_sizes, indices)
           #loss = -tf.reduce_mean(final_size_probs)
           
           loss = tf.reduce_mean(loss_fn(true_final_sizes,predicted_final_sizes))
           
        grads = tape.gradient(loss, prophet.trainable_variables)
        optimizer.apply_gradients(zip(grads, prophet.trainable_variables))   

        if episode % 10 == 0:
            print('Episode: ' + str(episode) + '; Loss: ' + f'{loss.numpy():.3f}')

        episode_losses.append(loss.numpy())
        
    return episode_losses

def build_demon(n,coding_dim=10,hard=True,summary=True):
    
    """
        Build demon model as variational autoencoder
         n (int): network node size
         coding_dim (int): dimension of latent space encoding
         summary (boolean): Print summary of keras model architecture
         hard (Boolean): If True reconstructed adj matrix is given as binary random variables, if False as probabilities.
         
    """

    """
        Set up the demon encoder model
    """
    inputs = keras.layers.Input(shape=[n,n])
    z = keras.layers.Flatten()(inputs)
    z = keras.layers.Dense(150,activation="selu")(z)
    z = keras.layers.Dense(100,activation="selu")(z)
    codings_mean = keras.layers.Dense(coding_dim)(z) # mean coding
    codings_log_var = keras.layers.Dense(coding_dim)(z) # log var coding
    codings = CodingSampler()([codings_mean,codings_log_var])
    demon_encoder = keras.Model(inputs=[inputs], outputs=[codings_mean, codings_log_var, codings])
    #demon_encoder.summary()
    
    """
        Set up the demon decoder model
        Demon outputs an n x n set of probabilities that each pair is connected  
    """
    decoder_inputs = keras.layers.Input(shape=[coding_dim])
    x = keras.layers.Dense(100,activation="selu")(decoder_inputs)
    x = keras.layers.Dense(150,activation="selu")(x)
    x = keras.layers.Dense(n * n,activation="sigmoid")(x)
    outputs = keras.layers.Reshape([n,n])(x)
    if hard:
        outputs = keras.layers.Lambda(lambda x: tf.nn.relu(tf.sign(x - tf.random.uniform(shape=[n,n], minval=0, maxval=1))))(outputs)
        #outputs = keras.layers.Lambda(lambda x: x + tf.random.uniform(shape=[n,n], minval=-0.01, maxval=0.01))(outputs) # test if this blocks backprop of gradients -- does not
        #outputs = keras.layers.Lambda(lambda x: tf.where(tf.random.uniform(shape=[n,n], minval=0, maxval=1) - x < 0, tf.ones([n,n]), tf.zeros([n,n])))(outputs) # this does seem to kill the gradients
    demon_decoder = keras.Model(inputs=[decoder_inputs],outputs=[outputs])
    #demon_decoder.summary()
    
    _, _, codings = demon_encoder(inputs)
    reconstructions = demon_decoder(codings)
    demon = keras.Model(inputs=[inputs], outputs=[codings_mean, codings_log_var, reconstructions])
    demon.summary()
    
    return demon

def train_demon(demon, train_nets, iterations=500):
    
    """
        Train prophet model on training set
    """
    
    """
        Set training params
    """
    batch_size = 1
    train_size = len(train_nets)
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
    episode_rec_losses = []
    for episode in range(iterations):
        
        # Sample realizations for training batch
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
        episode_rec_losses.append(rec_loss.numpy())
        
    return episode_losses, episode_rec_losses

def generate(sim,p_vals):

    """
        Generate training set of nets and final sizes
        Networks are simulated under an Erdos-Renyi model for now
            sim (GllespieSim): epi simulation model object
            p_vals (array-like): list of edge probabilities used to simulate ER networks (one per sim)
    """
    
    final_sizes = []
    nets = []
    for p in p_vals:
        edge_p = p
        G = nx.gnp_random_graph(n, edge_p) # Generate a random Erdos-Renyi graph with contact prob p
        adj = nx.to_numpy_array(G) # adjacency matrix
        sim.beta = adj
        #fsd = sim.approx_final_size_dist() # might be overally complicated for right now
        fsd = sim.mean_final_size() / n # compute mean final size
        final_sizes.append(fsd)
        nets.append(adj[:,:,np.newaxis])
        
    return np.array(final_sizes), np.array(nets)

def generate_demon_sizes(demon_probs):
    
    """
        Simulate final sizes from demon-generated networks
    """
    
    diffs = demon_probs - np.random.uniform(low=0.0, high=1.0, size=(batch_size,n,n))
    demon_nets = np.where(diffs < 0, 0, 1)
    dnet_final_sizes = []
    for idx, d_net in enumerate(demon_nets):
        demon_adj = np.triu(d_net) + np.triu(d_net).T # enforce symmetric adj requirment
        sim.beta = demon_adj
        fsd = sim.mean_final_size() / n # compute mean final size
        dnet_final_sizes.append(fsd) # normalized by pop size
        demon_nets[idx] = demon_adj
    
    return np.array(dnet_final_sizes), demon_nets # or convert to tensor?

# Sim params
n = 100 # pop size
final_time = 10.0 # time simulations should end
init_I = np.zeros(n)
init_I[0] = 1 # seed first infection
edge_p = .05  # edge probability
omega = 0 # immune waining: set to zero for SIR / Inf for SIS
nu = 0.5 # recovery/removal rate

""" 
    Init stochastic simulation
    Variables not supplied as keyword args will default to their defined values in __init__
"""    
sim = PoissonBinomialModel(final_time=final_time,
                    init_I=init_I,
                    beta=None,
                    immunity=True, 
                    nu=nu,
                    time_step=0.01)

"""
    Generate training set of nets and final sizes
"""
batch_size = 10
p_vals = np.tile(np.linspace(0.02, 0.05, num=16),10)
train_final_sizes, train_nets = generate(sim, p_vals)

"""
    Set up prophet model:
    Pre-train or restore prophet to predict final sizes without demon
"""
# restore = True
# if restore:
#     prophet = keras.models.load_model("prophet_async_trained.tf")
# else:
#     prophet = build_prophet(n,batch_size,summary=True)
#     episode_losses = train_prophet(prophet, train_final_sizes, train_nets, iterations=1000)
#     plot_training(episode_losses,'Loss','prophet_pretrain_loss_by_episode.png')
#     prophet.save("prophet_async_trained.tf")

"""
    Set up demon model
    Pre-train demon model to reconstruct ER-like networks
    Note: need to save in 'tf' format to serialize custom layers: https://www.tensorflow.org/guide/keras/save_and_serialize#registering_the_custom_object for details.
"""
restore = False
if restore:
    demon = keras.models.load_model("demon_async_trained.tf")
else:
    demon = build_demon(n,coding_dim=10,summary=True)
    episode_losses, episode_rec_losses = train_demon(demon, train_nets, iterations=1000)
    plot_training(episode_losses,'Loss','demon_pretrain_loss_by_episode.png')
    plot_training(episode_rec_losses,'Loss','demon_reconstruction_loss_by_episode.png')
    demon.save("demon_async_trained.tf",save_format="tf")

"""
    Now train demon to trick prophet
"""

"""
    Set training params
"""
# batch_size = 10
# iterations = 1000
# train_size = len(train_nets)
# mse_loss_fn = keras.losses.mean_squared_error
# latent_loss_fn = latent_loss()
# reconstruction_loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
# optimizer = keras.optimizers.Adam(lr=1e-3)

# """
#     We expect that the reconstruction and delusion loss might be negatively correlated
#     So need to (over)weight delusion loss
# """
# delusion_weight = 5.0

# """
#     Train demon to reconstruct ER-generated nets
#     Note: if batch_size > 1, all losses will be computed as averages across training instances in batch
# """
# episode_losses = []
# latent_losses = []
# reconstruction_losses = []
# delusion_losses = []
# for episode in range(iterations):
    
#     # Sample realizations for training batch
#     batch_indices = np.random.choice(train_size, batch_size)
#     net_batch = tf.convert_to_tensor(train_nets[batch_indices])
    
#     # Compute original prediction error
#     true_final_sizes = tf.convert_to_tensor(train_final_sizes[batch_indices])
#     true_final_sizes = tf.reshape(true_final_sizes, [batch_size,1])
#     predicted_final_sizes = prophet(net_batch)
#     p_abs_errors = tf.reduce_mean(mse_loss_fn(true_final_sizes,predicted_final_sizes))
    
#     with tf.GradientTape() as tape:
#         codings_mean, codings_log_var, reconstructions = demon(net_batch)
#         lat_loss = latent_loss_fn(codings_mean, codings_log_var)
#         rec_loss = reconstruction_loss_fn(net_batch,reconstructions)
        
#         #Sim new final sizes on demon-generated networks
#         demon_probs = reconstructions.numpy()
#         dnet_final_sizes, demon_nets = generate_demon_sizes(demon_probs)
        
#         """
#             Think we need an intermediate layer here that simulates the epidemic
#             on the generated network so gradients can flow back to discriminator
#         """
        
#         # Compute prediction error on demon-generated ents
#         dnet_batch = tf.convert_to_tensor(demon_nets)
#         predicted_final_sizes = prophet(dnet_batch)
#         q_abs_errors = tf.reduce_mean(mse_loss_fn(dnet_final_sizes,predicted_final_sizes))
        
#         delusion_loss = delusion_weight * (p_abs_errors - q_abs_errors) # the more negative the better
        
#         loss = lat_loss + rec_loss + delusion_loss # latent loss plus reconstruction loss

#     grads = tape.gradient(loss, demon.trainable_variables)
#     optimizer.apply_gradients(zip(grads, demon.trainable_variables))   

#     if episode % 10 == 0:
#         print('Episode: ' + str(episode) + '; Loss: ' + f'{loss.numpy():.3f}' + '; Latent Loss: ' + f'{lat_loss.numpy():.3f}' + '; Reconstruction Loss: ' + f'{rec_loss.numpy():.3f}' + '; Delusion Loss: ' + f'{delusion_loss.numpy():.3f}')

#     episode_losses.append(loss.numpy())
#     latent_losses.append(lat_loss.numpy())
#     reconstruction_losses.append(rec_loss.numpy())
#     delusion_losses.append(delusion_loss.numpy())
    
# plot_training(episode_losses,'Loss','demon_loss_by_episode.png')
# plot_training(latent_losses,'Loss','demon_latent_loss_by_episode.png')
# plot_training(reconstruction_losses,'Loss','demon_reconstruction_loss_by_episode.png')
# plot_training(delusion_losses,'Loss','demon_delusion_loss_by_episode.png')

# """
#     Compare statistical properties of demon-generated vs. ER-generated nets
# """
# test_size = 100
# rand_codings = tf.random.normal(shape=[test_size,coding_dim])
# demon_probs = demon_decoder(rand_codings).numpy()
# diffs = demon_probs - np.random.uniform(low=0.0, high=1.0, size=(test_size,n,n))
# demon_test_nets = np.where(diffs < 0, 0, 1)
    

# p_vals = np.array([0.03]*test_size)
# ER_test_nets = []
# for edge_p in p_vals:
#     G = nx.gnp_random_graph(n, edge_p)
#     ER_test_nets.append(G)

# """ 
#     Init stochastic simulation
#     Variables not supplied as keyword args will default to their defined values in __init__
# """
# from EpiNetSim import GillespieSim
# final_time = 10.0 # time simulations should end
# init_I = np.zeros(n)
# init_I[0] = 1 # seed first infection
# omega = 0 # immune waining: set to zero for SIR / Inf for SIS
# nu = 0.5 # recovery/removal rate   
# sim = GillespieSim(pops=n,
#                     terminate_condition="t[i] > 10",
#                     final_time=final_time,
#                     init_I=init_I,
#                     beta=None,
#                     omega=omega, 
#                     d=nu)

# """
#     Compare statistical properties (connectivity,degree distribution,clustering)
#     For more ideas see: https://networkx.org/documentation/stable/reference/algorithms/index.html
# """
# mean_degree = [] # mean of degree dist
# var_degree = [] # variance of degree dist
# num_edges = [] # number of edges
# clustering = [] # average clustering coefficient
# components = [] # number connected components
# diameter = [] # diameter - length of shortest path between most distant nodes
# final_size = []
# net_type = [] # type of network
# for demon_net, ER_net in zip(demon_test_nets, ER_test_nets):
    
#     demon_adj = np.triu(demon_net) + np.triu(demon_net).T # enforce symmetric adj requirment
#     demon_net = nx.from_numpy_matrix(demon_adj)
    
#     # Add stats for demon generated net
#     mean_degree.append(np.mean(nx.degree_histogram(demon_net)))
#     var_degree.append(np.var(nx.degree_histogram(demon_net)))
#     num_edges.append(demon_net.number_of_edges())
#     clustering.append(nx.average_clustering(demon_net))
#     components.append(nx.number_connected_components(demon_net))
#     if nx.is_connected(demon_net):
#         diameter.append(nx.diameter(demon_net))
#     else:
#         diameter.append(np.inf)
#     sim.beta = demon_adj
#     complete = False
#     while not complete: # make sure sims completes to final time
#         complete, f_size, times, I_traj = sim.run()
#     final_size.append(f_size / n) # normalized by pop size
#     net_type.append('Demon')
    
#     # Add stats for ER generated net
#     mean_degree.append(np.mean(nx.degree_histogram(ER_net)))
#     var_degree.append(np.var(nx.degree_histogram(ER_net)))
#     num_edges.append(ER_net.number_of_edges())
#     clustering.append(nx.average_clustering(ER_net))
#     components.append(nx.number_connected_components(ER_net))
#     if nx.is_connected(ER_net):
#         diameter.append(nx.diameter(ER_net))
#     else:
#         diameter.append(np.inf)
#     sim.beta = nx.to_numpy_array(ER_net)
#     complete = False
#     while not complete: # make sure sims completes to final time
#         complete, f_size, times, I_traj = sim.run()
#     final_size.append(f_size / n) # normalized by pop size
#     net_type.append('ER')

#     #print('Demon edges:' + str(demon_edges) + ' ;ER edges: ' + str(ER_edges))

# import pandas as pd
# test_dict = {'Mean Degree': mean_degree,
#              'Var Degree': var_degree,
#              'Num Edges': num_edges,
#              'Clustering': clustering,
#              'Components': components,
#              'Diameter': diameter,
#              'Final size': final_size,
#              'Net Type': net_type} 
# df = pd.DataFrame(test_dict)
# results_file = 'demon_net_reconstruction_stats.csv'
# df.to_csv(results_file,index=False) 
