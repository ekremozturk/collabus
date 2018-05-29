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
   
   
def plot_subset_metrics(raw_data, palette):

    df = pd.DataFrame(raw_data, columns = ['algo_no', 'kNN', 'Pop', 'MF'])
    # Setting the positions and width for the bars
    pos = list(range(len(df['MF']))) 
    width = 0.25 
    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10,5))
    
    # Create a bar with MF data,
    # in position pos,
    plt.bar(pos, 
            #using df['MF'] data,
            df['kNN'], 
            # of width
            width, 
            # with alpha 0.5
            alpha=0.8, 
            # with color
            color=palette[0], 
            # with label the first value in algo_no
            label=df['algo_no'][0]) 
    
    # Create a bar with kNN data,
    # in position pos + some width buffer,
    plt.bar([p + width for p in pos], 
            #using df['kNN'] data,
            df['Pop'],
            # of width
            width, 
            # with alpha 0.5
            alpha=0.8, 
            # with color
            color=palette[1], 
            # with label the second value in algo_no
            label=df['algo_no'][1]) 
    
    # Create a bar with Pop data,
    # in position pos + some width buffer,
    plt.bar([p + width*2 for p in pos], 
            #using df['Pop'] data,
            df['MF'], 
            # of width
            width, 
            # with alpha 0.5
            alpha=0.8, 
            # with color 7a0177
            color=palette[2], 
            # with label the third value in algo_no
            label=df['algo_no'][2]) 
    
        # Create a bar with Pop data,
    # in position pos + some width buffer,

    
    # Set the chart's title
    
    # Set the position of the x ticks
    ax.set_xticks([p + 1 * width for p in pos])
    
    # Set the labels for the x ticks
    ax.set_xticklabels(df['algo_no'])
    
    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos)-width, max(pos)+width*3)
    plt.ylim([0, 0.35] )
    # set 1 afterwards
    # Adding the legend and showing the plot
    plt.legend(['kNN', 'Pop', 'MF'], loc='upper left')
    plt.grid()
    plt.show()
###################################################################
subset1_4group_map = {'algo_no': ['MAP@20', 'MAP@50', 'MAP@200'],
    'kNN': [0.087681, 0.126087, 0.158637],
    'Pop': [0.0402027, 0.06431, 0.0748437],
    'MF': [0.160997, 0.260201,0.311535],

    }

subset2_4group_map = {'algo_no': ['MAP@20', 'MAP@50', 'MAP@200'],
    
    'kNN': [0.00159152, 0.00194432, 0.00255583],
    'Pop': [0.000982091, 0.00126922, 0.00184076],
    'MF': [0.00756854, 0.00923323,0.0113803]
    }

subset3_4group_map = {'algo_no': ['MAP@20', 'MAP@50', 'MAP@200'],
    
    'kNN': [0.0213323, 0.0253505, 0.0287871],
    'Pop': [0.0130185, 0.0154741, 0.0189481],
    'MF': [0.0567662, 0.0676903,0.0806219]
    }

rushmore = ['#0B775E','#35274A','#F2300F', '#E1BD6D']

plot_subset_metrics(subset1_4group_map,rushmore)
#=============================================================================

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
  
  return hist_y, user_y

def load_and_plot(subset_no):
  triplets, users, songs = fn.load_files_by_no(subset_no)
  userIDs, songIDs = fn.ids(users, songs)
  user_dict, song_dict = fn.form_dictionaries(userIDs, songIDs)
  triplets= fn.replace_DF(triplets, user_dict, song_dict)
  
  return prepare_plot(triplets, users, songs)
  
subset_no = ['1','2','3']
x = np.arange(5,201, 5)
hist_sets = list()
user_sets = list()
for no in subset_no:
  hist_y, user_y = load_and_plot(no)
  hist_sets.append(hist_y)
  user_sets.append(user_y)

for idx, y in enumerate(hist_sets):
  plt.plot(x,y ,color = rushmore [idx])
  plt.legend(subset_no)
  plt.xlabel("Number of Top Songs")
  plt.ylabel("Percentage")
  
for idx, y in enumerate(user_sets):
  plt.plot(x,y ,color = rushmore [idx])
  plt.legend(subset_no)
  plt.xlabel("Number of Top Songs")
  plt.ylabel("Percentage")

#hist_perc, user_perc = popularity_statistics(20)