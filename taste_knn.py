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
import song_details as sd

triplets, users, songs = fn.load_files()

#only ids
userIDs, songIDs = fn.ids(users, songs)

#dictionaries
user_dict, song_dict = fn.form_dictionaries(userIDs, songIDs)

train_DF, test_DF = fn.split_into_train_test(triplets)

start_time = time()
#train_DF, test_DF = fn.replace_DF(train_DF, test_DF, user_dict, song_dict)

#record and similarity matrices
R, M = fn.form_records(train_DF, user_dict, song_dict, normalization = True)
R_test, M_test = fn.form_records(test_DF, user_dict, song_dict, normalization = True)
elapsed_time = time()-start_time

def similar_items(songID, k=30):
  song_idx = song_dict[songID]
  song_array = pd.Series(M[song_idx,:]).sort_values(ascending=False)[1:k+1]
  indexes = song_array.index.values
  song_tuples = []
  for idx, song_sim in enumerate(song_array):
    song_tuples.append([indexes[idx],song_sim])
  return song_tuples

def k_similar_items(k=30):
  k_sim_songs = []
  for song in songIDs:
    k_sim_songs.append(similar_items(song, k))
  return k_sim_songs

K = np.asarray(k_similar_items(k=500))

def u_pred_i(userID, songID, k=30):
  sim_items = K[song_dict[songID]]
  sim_idx = sim_items[:,0].astype(int)
  sim_ratio = sim_items[:,1].astype(float)
  user_idx = user_dict[userID]
  R_user = R[user_idx, sim_idx].astype(float)
  
  return (sim_ratio*R_user).sum()/sim_ratio.sum()

def form_user_prediction(userID, k=30):
  user_idx = user_dict[userID]
  user_array = R[user_idx,:]
  user_pred = []
  for idx, item in enumerate(user_array):
    songID = songIDs[idx]
    if item == 0:
      r_pred = u_pred_i(userID, songID, k)
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
  user_pred = pd.Series(form_user_prediction(userID, 500)).sort_values(ascending=False)[0:n]
  indexes = user_pred.index.values
  return indexes

def rec_every_user(n=20):
  recommendations = []
  count = 0
  for _id in userIDs:
    recommendations.append(recommend_user(_id, n))
    count += 1
    print('User ' + str(count)+ ' finished -> ' + '%'+str(count/len(userIDs)*100)+' complete! ')
  return pd.DataFrame(data = recommendations, index=userIDs, columns=np.arange(n))

start_time = time()

#recommendations = rec_every_user(50)
recommendations = fn.rec_most_pop(userIDs, songs, by = 'occ', n=50)

_, ratings_eval = fn.form_tuples(R, R_test, knn=True)

ext_ratings_eval = fn.extract_evaluations(ratings_eval)

ext_recommendations = fn.extract_recommendations(recommendations, knn=True)

print("Preparing for metrics...")
pred_label = fn.prepare_prediction_label(ext_recommendations, ext_ratings_eval, knn=True)

from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .master("local[*]") \
    .appName("main")\
    .getOrCreate()

sc = spark.sparkContext
 
prediction_and_labels = sc.parallelize(pred_label)
metrics = RankingMetrics(prediction_and_labels)
mean_avg_prec= metrics.meanAveragePrecision
ndcg= metrics.precisionAt(20)

elapsed_time = time()-start_time

spark.stop()



#r = recommend_user('00106661302d2251d8bb661c91850caa65096457', 20)
#R_pred = form_R_pred(30)
#similar_items('SOAAFYH12A8C13717A' ,M, k=30)
#u_pred_i('00106661302d2251d8bb661c91850caa65096457', 'SOAAFYH12A8C13717A', M, k=30)
#form_user_prediction('00106661302d2251d8bb661c91850caa65096457', k=30)
