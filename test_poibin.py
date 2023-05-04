#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 07:22:54 2023

Test Poisson-Binomial prob mass function as implemented in poibin

@author: David
"""
from poibin import PoiBin
import numpy as np

n = 100 # trials
p = np.random.uniform(low=0.0, high=1.0, size=n) # prob success per trial
pb = PoiBin(p)

x = np.arange(0,n+1)
pr_x = pb.pmf(x) # compute prob mass Pr(K=x|p)

print(pr_x)
print(np.sum(pr_x))



