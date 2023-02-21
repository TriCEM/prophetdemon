#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 07:31:55 2023

Compare net statistics for ER vs. demon reconstructed networks

@author: David
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

results_file = '../demon_net_reconstruction_stats.csv'
df = pd.read_csv(results_file)

df['Std Dev Degree'] = df['Var Degree']**(1/2)
df['Edge Density'] = df['Num Edges'] / (100) # edge per node
df['Clustering'] = df['Clustering'] * 100 # transform to percent 

df = pd.melt(df, id_vars =['Net Type'],
             value_vars =['Mean Degree','Std Dev Degree', 'Edge Density', 'Clustering'])

# Draw a nested violinplot and split the violins for easier comparison
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
sns.violinplot(data=df, x="variable", y="value", hue="Net Type",
              split=True, linewidth=1, bw='silverman',
              palette={"Demon": "b", "ER": ".85"})

ax.set_xlabel('')
ax.set_ylabel('')

#legend_labels = ['Intrinsic','Predictive']
#ax.legend(labels=legend_labels) 

ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),prop={'size': 10})
plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
plt.xticks(fontsize=10)

fig.tight_layout()
png_file = 'demon_vs_ER_net_stats.png'
fig.savefig(png_file, dpi=200)