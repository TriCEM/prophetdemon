#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:00:16 2023

Implementation borrowed from: 
    https://blog.evjang.com/2016/11/tutorial-categorical-variational.html

@author: David
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

class GumbelSoftMax(keras.layers.Layer):
    
    def __init__(self,temperature,eps=1e-20,hard=True,**kwargs):
        
        """
        Args:
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        """
        
        self.temp = temperature
        
        self.eps = eps
        
        self.hard = hard
        
        self.n = 100
        
        super().__init__(**kwargs)
    
    def call(self,probs):
        
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        
        #probs = tf.reshape(probs, [self.n,1])
        #probs = tf.reshape(probs, [-1])
        one_minus_probs = 1.0 - probs
        logits = tf.math.log(tf.concat([probs, one_minus_probs], 0))
        
        logits = tf.transpose(logits)
        
        U = tf.random.uniform(tf.shape(logits), minval=0, maxval=1)
        sample = -tf.math.log(-tf.math.log(U + self.eps) + self.eps)
        
        y = logits + sample
        y = tf.nn.softmax( y / self.temp)
        
        if self.hard:
          #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),tf.shape(logits)[-1]), y.dtype)
          y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keepdims=True)),y.dtype)
          y = tf.stop_gradient(y_hard - y) + y # likely to prevent overflow, see: https://www.tensorflow.org/api_docs/python/tf/stop_gradient
        
        y = tf.reshape(y[:,0], [self.n,self.n]) # treat first column as Bernouli random variable
        
        return y
    
    
    def call_test_version(self,probs):
        
        """
            This version works with setup here in main
        """
        
        #probs = tf.reshape(probs, [probs.shape[-1],1])
        one_minus_probs = 1.0 - probs
        logits = tf.math.log(tf.concat([probs, one_minus_probs], 1))
        
        U = tf.random.uniform(tf.shape(logits), minval=0, maxval=1)
        sample = -tf.math.log(-tf.math.log(U + self.eps) + self.eps)
        
        y = logits + sample
        y = tf.nn.softmax( y / self.temp)
        
        if self.hard:
          #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),tf.shape(logits)[-1]), y.dtype)
          y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keepdims=True)),y.dtype)
          y = tf.stop_gradient(y_hard - y) + y
        
        y = tf.reshape(y[:,0], tf.shape(probs)) # treat first column as Bernouli random variable
        
        return y
               
def sample_gumbel(shape, eps=1e-20): 
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y

if __name__ == '__main__':

    n = 10
    temp = 0.1
    probs = np.random.uniform(low=0.0, high=1.0, size=[n,1])
    
    # Convert to TF tensor
    dtype = tf.float32 # data type (usually float32 or float64)
    probs = tf.constant(probs,dtype=dtype)
    
    # If transforming to logits
    #probs = np.concatenate([probs,1-probs],axis=1)
    #logits = np.log(probs)
    
    g = GumbelSoftMax(temp)
    
    import time
    tic = time.perf_counter()
    sample = g.call(probs)
    toc = time.perf_counter()
    elapsed = toc - tic

    print(f"Elapsed time: {elapsed:0.4f} seconds")
    print("Probs: \n" + str(probs))
    print("Returned sample: \n" + str(sample.numpy()))
