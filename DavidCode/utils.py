#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:40:56 2023

Utility functions for training prophet-demon models

@author: David
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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