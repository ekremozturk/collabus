#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 22:29:47 2018

@author: seha
"""

import numpy as np
import pandas as pd
from time import time
import taste_fn as fn
import matplotlib.pyplot as plt

# =============================================================================
# triplets, users, songs = fn.load_files()
# 
# songs.plot.scatter(x = 'occurence', y = 'totalPlay')
# 
# users.plot.scatter(x = 'occurence', y = 'totalPlay')
# 
# plt.hist(np.log(songs['occurence']), bins = 20, color = 'red')
# plt.xlabel('Occurence ln')
# plt.ylabel('Frequency')
# plt.show()
# =============================================================================



def plot_Rand(Rand_list):
   plt.plot(Rand_list)
   plt.title("Rand Plot")
   plt.show()
   
   
def plot_subset_metrics(raw_data):

    df = pd.DataFrame(raw_data, columns = ['algo_no', 'MF', 'kNN', 'Pop','Rand'])
    # Setting the positions and width for the bars
    pos = list(range(len(df['MF']))) 
    width = 0.2 
    print(df['Rand']    )
    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10,5))
    
    # Create a bar with MF data,
    # in position pos,
    plt.bar(pos, 
            #using df['MF'] data,
            df['MF'], 
            # of width
            width, 
            # with alpha 0.5
            alpha=0.8, 
            # with color
            color='#fbb4b9', 
            # with label the first value in algo_no
            label=df['algo_no'][0]) 
    
    # Create a bar with kNN data,
    # in position pos + some width buffer,
    plt.bar([p + width for p in pos], 
            #using df['kNN'] data,
            df['kNN'],
            # of width
            width, 
            # with alpha 0.5
            alpha=0.8, 
            # with color
            color='#f768a1', 
            # with label the second value in algo_no
            label=df['algo_no'][1]) 
    
    # Create a bar with Pop data,
    # in position pos + some width buffer,
    plt.bar([p + width*2 for p in pos], 
            #using df['Pop'] data,
            df['Pop'], 
            # of width
            width, 
            # with alpha 0.5
            alpha=0.8, 
            # with color 7a0177
            color='#c51b8a', 
            # with label the third value in algo_no
            label=df['algo_no'][2]) 
    
        # Create a bar with Pop data,
    # in position pos + some width buffer,
    plt.bar([p + width*3 for p in pos], 
            #using df['Pop'] data,
            df['Rand'], 
            # of width
            width, 
            # with alpha 0.5
            alpha=0.8, 
            # with color 7a0177
            color='#7a0177',
            # with label the third value in algo_no
            label=df['algo_no'][3]) 
    # Set the y axis label
    ax.set_ylabel('Score')
    
    # Set the chart's title
    ax.set_title('Metrics for Subset1')
    
    # Set the position of the x ticks
    ax.set_xticks([p + 1.5 * width for p in pos])
    
    # Set the labels for the x ticks
    ax.set_xticklabels(df['algo_no'])
    
    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos)-width, max(pos)+width*4)
    plt.ylim([0, 0.6] )
    # set 1 afterwards
    # Adding the legend and showing the plot
    plt.legend(['MF', 'kNN', 'Pop', 'Rand'], loc='upper left')
    plt.grid()
    plt.show()
###################################################################
raw_data_subset1 = {'algo_no': ['MAP@20', 'MAP@50', 'MAP@200', 'NDCG'],
    'MF': [0.258671859, 0.334881281,0.346189630, 0.111],
    'kNN': [0.409314041, 0.438968362, 0.446123075, 0.515524517],
    'Pop': [0.262060313, 0.338270778, 0.349557471, 0.380154275],
    'Rand': [0.002336661, 0.006184873, 0.0059568534, 0.02574817]}

plot_subset_metrics(raw_data_subset1)