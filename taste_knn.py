#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 11:23:24 2018

@author: ekrem
"""

import numpy as np
import pandas as pd
from time import time
import taste_fn as fn

triplets, users, songs = fn.load_files()

#only ids
userIDs, songIDs = fn.ids(users, songs)

#dictionaries
user_dict, song_dict = fn.form_dictionaries(userIDs, songIDs)

R, M = fn.form_records(triplets, user_dict, song_dict, normalization = True)

def similar_items(songID ,M, k=30):
  song_idx = song_dict[songID]
  song_array = pd.Series(M[song_idx,:]).sort_values(ascending=False)[1:k+1]
  indexes = song_array.index.values
  song_tuples = []
  for idx, song_sim in enumerate(song_array):
    song_tuples.append((song_sim, songIDs[indexes[idx]]))
  return song_tuples

def u_pred_i(userID, songID, M, k=30):
  sim_items = similar_items(songID, M, k)
  user_idx = user_dict[userID]
  sum_pred = 0
  w_i = 0
  for item in sim_items:
    id_ij = item[1]
    w_ij = item[0]
    w_i += w_ij
    idx_ij = song_dict[id_ij]
    r_uj = R[user_idx, idx_ij]
    sum_pred += w_ij*r_uj

  return sum_pred/w_i

def form_user_prediction(userID, k=30):
  user_idx = user_dict[userID]
  user_array = R[user_idx,:]
  user_pred = []
  for idx, item in enumerate(user_array):
    songID = songIDs[idx]
    if item == 0:
      r_pred = u_pred_i(userID, songID, M, k)
    else:
      r_pred = 0
    user_pred.append(r_pred)
  return user_pred

def form_R_pred(k):
  R_pred = []
  for uid in userIDs[0:1]:
    preds = form_user_prediction(uid, k)
    R_pred.append(preds)
  return pd.DataFrame(data = R_pred)

def recommend_user(userID, n=20):
  user_pred = pd.Series(form_user_prediction(userID, 30)).sort_values(ascending=False)[0:n]
  indexes = user_pred.index.values
  return songIDs[indexes]

def rec_every_user(n=20):
  recommendations = []
  for _id in userIDs:
    recommendations.append(recommend_user(_id, n))
  return pd.DataFrame(data = recommendations, index=userIDs, columns=np.arange(n))

start_time = time()
#r = recommend_user('00106661302d2251d8bb661c91850caa65096457', 20)
recommendations = rec_every_user(20)
#R_pred = form_R_pred(30)
#similar_items('SOAAFYH12A8C13717A' ,M, k=30)
#u_pred_i('00106661302d2251d8bb661c91850caa65096457', 'SOAAFYH12A8C13717A', M, k=30)
#form_user_prediction('00106661302d2251d8bb661c91850caa65096457', k=30)
elapsed_time = time()-start_time

#seha
# Precision metric kesin, daha sonra mean average precision
# Başka metric varsa onlar da
# Muhtemelen hepsi scikit gibi librarylerde vardır

#ekrem

#herkes
# msd h5 summary


