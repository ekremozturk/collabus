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

#=============================================================================

start_time = time()

def popularity_statistics(first_n, triplets, users, songs):
  songs_by_occ = songs.sort_values(by='occurence', ascending=False)[:first_n]
  idx_by_occ = songs_by_occ.index.values
  triplets_n = triplets.loc[triplets['itemID'].isin(idx_by_occ)]
  hist_perc = len(triplets_n)/len(triplets)*100
  user_perc = len(triplets_n.groupby('userID'))/len(users)*100
  return hist_perc, user_perc

def prepare_plot(triplets, users, songs):
  points = np.arange(5,201, 5)
  hist_y = list()
  user_y = list()
  for point in points:
    hist_perc, user_perc = popularity_statistics(point, triplets, users, songs)
    hist_y.append(hist_perc)
    user_y.append(user_perc)
    print('Point', point, 'finished!')
  
  return hist_y, user_y

def load_and_plot(subset_no):
  print('Loading', subset_no)
  triplets, users, songs = fn.load_files_by_no(subset_no)
  userIDs, songIDs = fn.ids(users, songs)
  user_dict, song_dict = fn.form_dictionaries(userIDs, songIDs)
  print("Replacing IDs with indexes....")
  triplets= fn.replace_DF(triplets, user_dict, song_dict)
  print('Finished loading', subset_no)
  return prepare_plot(triplets, users, songs)
  
subset_no = ['1','2','3', '-original']

x = np.arange(5,201, 5)
hist_sets = list()
user_sets = list()
for no in subset_no:
  hist_y, user_y = load_and_plot(no)
  hist_sets.append(hist_y)
  user_sets.append(user_y)
  print('Subset', no,'finished!')

elapsed_time = time()-start_time

for y in hist_sets:
  plt.plot(x,y)
  plt.legend(subset_no)
  
for y in user_sets:
  plt.plot(x,y)
  plt.legend(subset_no)