#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:07:42 2023

@author: david
"""

import tensorflow_probability as tfp
#from tfp.distributions import RelaxedBernoulli

temperature = 0.5
p = [0.1, 0.5, 0.4]
dist = tfp.distributions.RelaxedBernoulli(temperature, probs=p)
sample = dist.sample((2,))
print(sample)