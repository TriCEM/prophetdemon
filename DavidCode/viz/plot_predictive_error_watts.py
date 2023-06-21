#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 07:31:55 2023

Plot predictive error in final sizes versus stochastic noise/error in final sizes

@author: David
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

results_file = 'prophet_finalsize_predictions_watts.csv'
df = pd.read_csv(results_file)

# Compute predictive error
df['Predictive Error'] = df['Test Final Size'] - df['Predicted Final Size']

# Compute stochastic deviation from mean at each edge prob
mean_size_by_rewire_p = df.groupby('Rewiring Prob').mean()['Test Final Size']
df['Mean Size'] = [mean_size_by_rewire_p[x] for x in df['Rewiring Prob']]
df['Stochastic Error'] = df['Test Final Size'] - df['Mean Size']

df = pd.melt(df, id_vars =['Rewiring Prob'],
             value_vars =['Stochastic Error', 'Predictive Error'],
             var_name ='Error Type',
             value_name ='Final Size Error')

# Draw a nested violinplot and split the violins for easier comparison
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
sns.violinplot(data=df, x="Rewiring Prob", y="Final Size Error", hue="Error Type",
              split=True, linewidth=1, bw='silverman',
              palette={"Stochastic Error": "b", "Predictive Error": ".85"})

ax.set_xticklabels([f'{x:.2f}' for x in df['Rewiring Prob'].unique()])
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),prop={'size': 10})

fig.tight_layout()
png_file = 'prophet_finalsize_deviation_by_rewiringprobs.png'
fig.savefig(png_file, dpi=200)