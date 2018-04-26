#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 11:23:24 2018

@author: ekrem
"""

import numpy as np
import pandas as pd
import taste_main_script as fn
from math import log

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
def kNN_build(triplets, userIDs, songIDs):

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

R, M = kNN_build(triplets, userIDs, songIDs)

def similarItems(ID ,M, k):
  song_idx = song_dict[ID]
  song_array = M[song_idx,:]
  song_tuples = []
  for song_sim,song_id in zip(song_array,songIDs):
    if(song_sim != 0):
      song_tuples.append((song_sim, song_id))
  song_tuples = sorted(song_tuples, key=lambda x: x[0], reverse=True)
  return song_tuples[1:k+1]

tuples = similarItems('SOAAFYH12A8C13717A', M, 30)