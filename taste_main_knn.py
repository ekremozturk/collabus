#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 11:23:24 2018

@author: ekrem
"""

import numpy as np
import pandas as pd
import taste_main_script as fn

from math import log, exp
from numpy.linalg import norm

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

#only ids
userIDs = np.asarray(users['ID'])
songIDs = np.asarray(songs['ID'])

##############################################################
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
##############################################################

user_dict, song_dict = form_dictionaries(userIDs, songIDs)

##############################################################
def kNN_form(triplets, userIDs, songIDs):

  #Countların logaritması güzel normalization oluyormuş
  #1 countların logaritması 0 çıkacağı ve çarpımlarda sorun olacağı için için 1 ekledim
  triplets_logged = triplets['playCount'].apply(log)+1

  #Bu R matrix'i user-item
  #Direk logaritmaları ekledim buraya
  R = np.zeros((userIDs.size, songIDs.size))
  for t, lc in zip(np.asmatrix(triplets), triplets_logged):
    user_idx = user_dict[t[0,0]]
    song_idx = song_dict[t[0,1]]
    R[user_idx, song_idx] = lc

  #Bu mucizevi bir şey
  #Emin değilim ama diagonal 1 çıktığına göre doğrudur hesabı
  #Transpose'u alınca item, almadan soksam user simi buluyor
  #User simi yanlış hesaplıyor olabilir çünkü benim bildiğim implicit feedbackte user sim hesaplayamıyorsun
  from sklearn.metrics.pairwise import cosine_similarity
  M = cosine_similarity(R.transpose())

  return R, M
##############################################################

R, M = kNN_form(triplets, userIDs, songIDs)

def similar_items(songID ,M, k):
  song_idx = song_dict[songID]
  song_array = M[song_idx,:]
  song_tuples = []
  for song_sim,song_id in zip(song_array,songIDs):
    if(song_sim != 0):
      song_tuples.append((song_sim, song_id))
  song_tuples = sorted(song_tuples, key=lambda x: x[0], reverse=True)
  return song_tuples[1:k+1]

tuples = similar_items('SOAAFYH12A8C13717A', M, 30) 

def u_pred_i(userID, songID, M, k):
  sim_items = similar_items(songID, M, k)
  user_idx = user_dict[userID]
  sum_pred = 0
  w_i = []
  for item in sim_items:
    id_ij = item[1]
    w_ij = item[0]
    w_i.append(w_ij)
    idx_ij = song_dict[id_ij]
    r_uj = R[user_idx, idx_ij]
    sum_pred += w_ij*r_uj
  
  return sum_pred/norm(w_i)

def form_predictions(userID):
  R_pred = R
  
k = exp(u_pred_i('00106661302d2251d8bb661c91850caa65096457', 'SOCDDDO12AC4689F56', M, 30)-1)  
  

#seha
# Precision metric kesin, daha sonra mean average precision
# Başka metric varsa onlar da
# Muhtemelen hepsi scikit gibi librarylerde vardır

#ekrem
# Diğer scriptteki fonksiyonlar pandas dataframelere göre temize çekilecek
# Pandas to pyspark dataframe çevirisi MF için yapılacak ve MF yapılacak daha temiz
# MF parametrelerini ayarlamak için cross validation

#herkes
# Evaluation için de cross validation
# gridsearchcv 
  
  
  
  