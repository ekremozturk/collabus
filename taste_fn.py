#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 17:12:17 2018

@author: ekrem
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from math import log, exp

##############################################################################

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

##############################################################################

def ids(users, songs):
  userIDs = np.asarray(users['ID'])
  songIDs = np.asarray(songs['ID'])
  return userIDs, songIDs

##############################################################################

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

##############################################################################

def split_into_train_test(df, frac=0.8):
  train = df.groupby("userID", group_keys=False).apply(lambda df: df.sample(frac=frac))
  test = df.drop(train.index)
  return train, test

##############################################################################

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
  
##############################################################################
    
def form_tuples(record_train, record_test, virtual=False):

  print("Creating rating tuples...")
  
  if virtual:
    ratings_train = []
    for i in range(record_train.shape[0]):
      for j in range(record_train.shape[1]):
        count = float(record_train.iloc[i,j])
        rating = (i, j, count)
        ratings_train.append(rating)
  
    ratings_test = []
    ratings_eval = []
    for i in range(record_test.shape[0]):
      for j in range(record_test.shape[1]):
        count = record_test.iloc[i,j]
        if(count>0):
          rating = (i, j)
          evali = (i, j, 1.0)
          ratings_test.append(rating)
          ratings_eval.append(evali)
  
    return ratings_train, ratings_eval

  ratings_train = []
  for i in range(record_train.shape[0]):
    for j in range(record_train.shape[1]):
      count = float(record_train[i,j])
      rating = (i, j, count)
      ratings_train.append(rating)

  ratings_test = []
  ratings_eval = []
  for i in range(record_test.shape[0]):
    for j in range(record_test.shape[1]):
      count = record_test[i,j]
      if(count>0):
        rating = (i, j)
        evali = (i, j, 1.0)
        ratings_test.append(rating)
        ratings_eval.append(evali)

  return ratings_train, ratings_eval

##############################################################################
  
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
  
##############################################################################
  
def extract_recommendations(recommendations, knn=False):
  rec = defaultdict(list)
  
  if knn:
    index = 0
    for idx, user in recommendations.iterrows():
      list_of_songs = list(user)
      for song in list_of_songs:
        rec[index].append(song)
      index += 1
    return rec

  for row in recommendations:
      user_no = row.user
      for recommend in row.recommendations:
          rec[user_no].append(recommend.item)
            


def extract_evaluations(ratings_eval):
    eval_dict = defaultdict(list)
    for row in ratings_eval:
        eval_dict[row[0]].append(row[1])
    return eval_dict

def prepare_prediction_label(recommendations, ratings, knn):
    
  if knn:
      tuples = []
      for song, recommend in recommendations.items():
          tuples.append((recommend,ratings[song]))
      return tuples
    
  recommend_ext = extract_recommendations(recommendations)
  rating_ext = extract_evaluations(ratings)
  tuples = []
  for song, recommend in recommend_ext.items():
      tuples.append((recommend,rating_ext[song]))
  return tuples
    
##############################################################################

def group_users(userIDs, g_size):
  
  np.random.shuffle(userIDs)
  
  groups = []
  
  remaining_group = userIDs.size % g_size
  
  for i in range(0, userIDs.size - remaining_group, g_size):
    group = []
    for j in range(0, g_size):
      group.append(userIDs[i+j])
      
    groups.append(group)
      
  group = []  
  for i in range(remaining_group):
    group.append(userIDs[-i])
  
  groups.append(group)
  
  return groups
  
##############################################################################

def agg_fn(agg, item):
  if(agg=='average'):
    return item.mean()
  if(agg=='normalized_avg'):
    item = item+1
    item = item.apply(log)+1
    return exp(item.mean())

def form_groups(userGroups, train_data, test_data):
  merged_train_ratings = []
  merged_test_ratings = []
  for group in userGroups:
    train_group_Series = []
    test_group_Series = []
    for user in group:
      single_train_data = train_data[train_data['userID']==user]
      single_train_Series = pd.Series(list(single_train_data['playCount']), index = single_train_data['itemID'], name=user)
      train_group_Series.append(single_train_Series)
      
      single_test_data = test_data[test_data['userID']==user]
      single_test_Series = pd.Series(list(single_test_data['playCount']), index = single_test_data['itemID'], name=user)
      test_group_Series.append(single_test_Series)
      
    merged_train_ratings.append(pd.concat(train_group_Series, axis=1).fillna(0).astype(int))
    merged_test_ratings.append(pd.concat(test_group_Series, axis=1).fillna(0).astype(int))

  return merged_train_ratings, merged_test_ratings

def form_virtual_users(groups, song_dict, agg = 'average'):
  virtual_users = []
  for group in groups:
    virtual_user = []
    for idx, item in group.iterrows():
      virtual_user.append(agg_fn(agg, item))
    virtual_users.append(pd.Series(virtual_user, index = group.index.values).fillna(0))
  virtual_users = pd.DataFrame(virtual_users).fillna(0)
  song_idx_cols = pd.Series([song_dict[x] for x in virtual_users.columns.values], index = virtual_users.columns.values)
  
  return virtual_users.rename(columns=song_idx_cols)      
  
##############################################################################
  
def extract_most_pop(songs, n):
  
  by_tot_play= songs.sort_values('totalPlay', ascending=False).iloc[:n].index.values
  by_occurence= songs.sort_values('occurence', ascending=False).iloc[:n].index.values
  by_mean= songs.sort_values('mean', ascending=False).iloc[:n].index.values
  
  return by_tot_play, by_occurence, by_mean
  
def rec_most_pop(userIDs, songs,  by = 'occ', n=20):
  
  totPlay, occ, mean = extract_most_pop(songs, n)
  
  if by == 'tot':
    return pd.DataFrame(np.full((len(userIDs), n), totPlay, dtype=int), index=userIDs)
  elif by == 'occ':
    return pd.DataFrame(np.full((len(userIDs), n), occ, dtype=int), index=userIDs)
  elif by == 'mean':
    return pd.DataFrame(np.full((len(userIDs), n), mean, dtype=int), index=userIDs)
  