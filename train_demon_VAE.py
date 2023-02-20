#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Train demon model to generate contact networks using a variational autoencoder (VAE)

For now just testing to see if the demon VAE can generate networks similar to random graphs

TODO: Enforce requirment that demon generated adjaceny matrices are symmetric
    by setting upper diagonal equal to lower diagonal:
        m = np.tril(a) + np.tril(a).T

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
train_size = 10 
n=28 # num nodes in contact network
p_vals = np.array([0.025]*train_size)
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

_, _, codings = demon_encoder(inputs)
reconstructions = demon_decoder(codings)
demon = keras.Model(inputs=[inputs], outputs=[codings_mean, codings_log_var, reconstructions])
demon.summary()

"""
    Set training params
"""
batch_size = 1
epochs = 250
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

    #if episode % 10 == 0:
    print('Episode: ' + str(episode) + '; Loss: ' + f'{loss.numpy():.3f}' + '; Latent Loss: ' + f'{lat_loss.numpy():.3f}' + '; Reconstruction Loss: ' + f'{rec_loss.numpy():.3f}')

    episode_losses.append(loss.numpy())

plot_training(episode_losses,'Loss','loss_by_episode.png')

"""
    How do we test "goodness" of reconstructed nets generated by demon?
        -Compare statistical properties (mean_connectivity,degree distribution,clustering)
        -Compare final size distributions on demon nets vs. ER nets
"""

"""
    Compare statistical properties of demon-generated vs. ER-generated nets
"""
test_size = 10
rand_codings = tf.random.normal(shape=[test_size,coding_dim])
demon_probs = demon_decoder(rand_codings).numpy()
diffs = demon_probs - np.random.uniform(low=0.0, high=1.0, size=(test_size,n,n))
demon_test_nets = np.where(diffs < 0, 0, 1)

p_vals = np.array([0.025]*test_size)
ER_test_nets = []
for edge_p in p_vals:
    G = nx.gnp_random_graph(n, edge_p)
    ER_test_nets.append(G)

for demon_net, ER_net in zip(demon_test_nets, ER_test_nets):
    demon_net=nx.from_numpy_matrix(demon_net)
    demon_edges = demon_net.number_of_edges()
    ER_edges = ER_net.number_of_edges()
    
    print('Demon edges:' + str(demon_edges) + ' ;ER edges: ' + str(ER_edges))
      

