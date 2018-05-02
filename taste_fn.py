#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 17:12:17 2018

@author: ekrem
"""
import numpy as np
import pandas as pd
from math import log

def load_files():
  triplets = pd.read_table('subset/train_triplets_echonest.txt',
                         sep=' ',
                         header=None,
                         names=['userID','itemID','playCount'])

  users = pd.read_table('subset/echonest_user_play_mean.txt',
                         sep=' ',
                         header=None,
                         names=['ID','totalPlay','occurence', 'mean'])

  songs = pd.read_table('subset/echonest_song_play_mean.txt',
                         sep=' ',
                         header=None,
                         names=['ID','totalPlay','occurence', 'mean'])
  
  return triplets, users, songs

def ids(users, songs):
  userIDs = np.asarray(users['ID'])
  songIDs = np.asarray(songs['ID'])
  return userIDs, songIDs

def form_dictionaries(userIDs, songIDs):
  song_dict = dict()
  user_dict = dict()

  count = 0
  for item in songIDs:
      song_dict[item] = count
      count += 1

  count = 0
  for item in userIDs:
      user_dict[item] = count
      count += 1

  return user_dict, song_dict

def split_into_train_test(df, frac=0.8):
  train = df.groupby("userID", group_keys=False).apply(lambda df: df.sample(frac=frac))
  test = df.drop(train.index)
  return train, test

def form_records(triplets, user_dict, song_dict, normalization = False):
  
  R = np.zeros((len(user_dict), len(song_dict)))
  
  if normalization:
    
    #Log(playCount)+1
    counts_logged = triplets['playCount'].apply(log)+1
    
    for t, logged_count in zip(np.asmatrix(triplets), counts_logged):
      user_idx = user_dict[t[0,0]]
      song_idx = song_dict[t[0,1]]
      R[user_idx, song_idx] = logged_count
  
    #Form item-item similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    M = cosine_similarity(R.transpose())
  
    return R, M
  
  else:
    
    for t in triplets.values:
      user_idx = user_dict[t[0]]
      song_idx = song_dict[t[1]]
      R[user_idx, song_idx] = t[2]
      
    return R
  
def get_subsets():
  
  train_triplets = pd.read_table('subset/train_triplets.txt',
                         sep=' ',
                         header=None,
                         names=['userID','itemID','playCount'])
  
  test_triplets = pd.read_table('subset/test_triplets.txt',
                       sep=' ',
                       header=None,
                       names=['userID','itemID','playCount'])

  return train_triplets, test_triplets
  
  
  
  
  
  
  
  