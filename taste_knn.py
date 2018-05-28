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

start_time = time()

print('Loading..')
triplets, users, songs = fn.load_files()
userIDs, songIDs = fn.ids(users, songs) #only ids
user_dict, song_dict = fn.form_dictionaries(userIDs, songIDs) #dictionaries

#=============================================================================

#triplets = fn.replace_DF(triplets, user_dict, song_dict)
#print('Splitting into train and test sets..')
#train_DF, test_DF = fn.split_into_train_test(triplets, frac=0.5)
#print('Forming record and similarity matrices..')
#R, M = fn.form_records(train_DF, user_dict, song_dict, normalization = True)
#R_test, _ = fn.form_records(test_DF, user_dict, song_dict, normalization = True)

#=============================================================================

print("Forming virtual users....")
train_groups, test_groups = fn.load_groups(12)
groups_weights = fn.form_group_weights(train_groups, user_dict, users)
virtual_training = fn.form_virtual_users(train_groups, song_dict, agg='normalized_avg')
virtual_test = fn.form_virtual_users(test_groups, song_dict, agg='normalized_avg')
R, M = fn.form_records(virtual_training, user_dict, song_dict, normalization = True, virtual=True)
R_test, _ = fn.form_records(virtual_test, user_dict, song_dict, normalization = True, virtual=True)

#=============================================================================

elapsed_time_form = time()-start_time

#=============================================================================

def similar_items(songID, k=30):
  song_idx = song_dict[songID]
  song_array = pd.Series(M[song_idx,:]).sort_values(ascending=False)[1:k+1]
  indexes = song_array.index.values
  song_tuples = []
  for idx, song_sim in enumerate(song_array):
    song_tuples.append([indexes[idx],song_sim])
  return song_tuples

def k_similar_items(k=50):
  k_sim_songs = []
  for song in songIDs:
    k_sim_songs.append(similar_items(song, k))
  return k_sim_songs

K = np.asarray(k_similar_items(k=50))

def u_pred_i(user_idx, songID, k=50):
  sim_items = K[song_dict[songID]]
  sim_idx = sim_items[:,0].astype(int)
  sim_ratio = sim_items[:,1].astype(float)
  R_user = R[user_idx, sim_idx].astype(float)
  
  return (sim_ratio*R_user).sum()/sim_ratio.sum()

def form_user_prediction(user_idx, k=50):
  user_array = R[user_idx,:]
  user_pred = []
  for idx, item in enumerate(user_array):
    songID = songIDs[idx]
    if item == 0:
      r_pred = u_pred_i(user_idx, songID, k)
    else:
      r_pred = 0
    user_pred.append(r_pred)
  return user_pred

def recommend_user(user_idx, n=20):
  user_pred = pd.Series(form_user_prediction(user_idx, 50)).sort_values(ascending=False)[0:n]
  indexes = user_pred.index.values
  return indexes

def rec_every_user(n=20):
  recommendations = []
  count = 0
  for user_idx, value in enumerate(R):
    recommendations.append(recommend_user(user_idx, n))
    count += 1
    if count%100 == 0:
      print('User ' + str(count)+ ' finished -> ' + '%'+str(count/len(R)*100)+' complete! ')
  return pd.DataFrame(data = recommendations, index=np.arange(len(R)), columns=np.arange(n))

#=============================================================================

def evaluate(pred_label):
  f1_, precision_, recall_ = fn.f1_precision_recall(pred_label)
  mpr_ = fn.mpr(pred_label)
  prediction_and_labels = sc.parallelize(pred_label)
  metrics = RankingMetrics(prediction_and_labels)
  map_= metrics.meanAveragePrecision
  ndcg_ = metrics.ndcgAt(10)
  return map_, ndcg_, f1_, precision_, recall_, mpr_

def evaluate_knn(ext_ratings_eval):
  scores = list()
  
  for n_ in fn.n_list:
    recommendations = rec_every_user(n=n_)
    ext_recommendations = fn.extract_recommendations(recommendations, knn=True)
    print("Preparing for metrics...", n_)
    pred_label = fn.prepare_prediction_label(ext_recommendations, ext_ratings_eval, knn=True)
    scores.append(evaluate(pred_label))

  scores = pd.DataFrame(scores, index = [fn.n_list], columns=['mAP', 'NDCG', 'F1', 'Precision', 'Recall', 'mPR'])
  return scores
  
def evaluate_pop(ext_ratings_eval):
  scores = list()
  
  for n_ in fn.n_list:
    recommendations = fn.rec_most_pop(users, songs, by = 'occ', n=n_)
    ext_recommendations = fn.extract_recommendations(recommendations, knn=True)
    print("Preparing for metrics...", n_)
    pred_label = fn.prepare_prediction_label(ext_recommendations, ext_ratings_eval, knn=True)
    scores.append(evaluate(pred_label))
    
  scores = pd.DataFrame(scores, index = [fn.n_list], columns=['mAP', 'NDCG', 'F1', 'Precision', 'Recall', 'mPR'])
  return scores


def evaluate_rand(ext_ratings_eval): 
  scores = list()
  
  for n_ in fn.n_list:
    recommendations = fn.rec_random(R, songs, n=n_)
    ext_recommendations = fn.extract_recommendations(recommendations, knn=True)
    print("Preparing for metrics...", n_)
    pred_label = fn.prepare_prediction_label(ext_recommendations, ext_ratings_eval, knn=True)
    scores.append(evaluate(pred_label))
    
  scores = pd.DataFrame(scores, index = [fn.n_list], columns=['mAP', 'NDCG', 'F1', 'Precision', 'Recall', 'mPR'])
  return scores

#=============================================================================

#_, ratings_eval = fn.form_tuples(train_DF, test_DF)
_, ratings_eval = fn.form_tuples(virtual_training, virtual_test, virtual=True)
ext_ratings_eval = fn.extract_evaluations(ratings_eval)

start_time = time()

from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .master("local[*]") \
    .appName("main")\
    .getOrCreate()

sc = spark.sparkContext

scores_knn = evaluate_knn(ext_ratings_eval)
scores_pop = evaluate_pop(ext_ratings_eval)
scores_rand = evaluate_rand(ext_ratings_eval)

spark.stop()

elapsed_time = time()-start_time