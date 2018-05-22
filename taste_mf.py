#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:07:13 2018

@author: ekrem
    """

from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.mllib.recommendation import ALS
from time import time
import pandas as pd
import numpy as np
import taste_fn as fn

triplets, users, songs = fn.load_files()
userIDs, songIDs = fn.ids(users, songs)
user_dict, song_dict = fn.form_dictionaries(userIDs, songIDs)

print("Replacing IDs with indexes....")
triplets= fn.replace_DF(triplets, user_dict, song_dict)

##############################################################################
print("Splitting into sets....")
train_DF, test_DF = fn.split_into_train_test(triplets, frac=0.5)

##############################################################################
print("Splitting into subsets....")
subsets = fn.split_into_train_test_cv(triplets, cv=3)

##############################################################################
def load_user_groups(song_dict, group_size=4, agg = 'normalized_avg'):
  print("Forming user groups....")
  train_groups, test_groups = fn.load_groups(group_size)
  virtual_training = fn.form_virtual_users(train_groups, song_dict, agg=agg)
  virtual_test = fn.form_virtual_users(test_groups, song_dict, agg=agg)
  
  return virtual_training, virtual_test

##############################################################################
def train(train_rdd, params):
  
  print('Training...')
  rank_, lambda_, alpha_ = params[0], params[1], params[2]
  model = ALS.trainImplicit(train_rdd, 
                            rank_, 
                            iterations=10, 
                            lambda_=lambda_, 
                            alpha=alpha_,
                            seed=10)
  
  return model

##############################################################################
def evaluate(model, test_set, n=20):
  print("Making recommendations...")
  recommendations = model.recommendProductsForUsers(n).collect()
  
  print("Preparing for metrics...")
  pred_label = fn.prepare_prediction_label(recommendations,test_set)
  prediction_and_labels = sc.parallelize(pred_label)
  metrics = RankingMetrics(prediction_and_labels)

  return metrics

############################################################################## 
def cross_validation(subsets, paramGrid, n=200):
  cv_scores = list()
  map_scores = np.zeros((len(subsets), len(paramGrid)))
  ndcg_scores = np.zeros((len(subsets), len(paramGrid)))
  map_statistics = np.zeros((3, len(paramGrid)))
  ndcg_statistics = np.zeros((3, len(paramGrid)))
  
  for num in range (len(subsets)):
    print("Creating train sets and test set for trial ", num+1)
    
    test_DF = subsets[num]
    train_DF = pd.concat([element for i, element in enumerate(subsets) if i not in {num}])
    train_rdd, test_set = form_and_rdd(train_DF, test_DF)
    
    for param_idx, params in enumerate(paramGrid):
      model = train(train_rdd, params)
      metrics = evaluate(model, test_set, n)
      map_, ndcg_ = metrics.meanAveragePrecision, metrics.ndcgAt(10)
      map_scores[num, param_idx] = map_
      ndcg_scores[num, param_idx] = ndcg_
      print("Completed: " + str(num*len(paramGrid)+param_idx))
  
  map_statistics[0,:] = map_scores.mean(axis=0)
  ndcg_statistics[0,:] = ndcg_scores.mean(axis=0)
  map_statistics[1,:] = map_scores.min(axis=0)
  ndcg_statistics[1,:] = ndcg_scores.min(axis=0)
  map_statistics[2,:] = map_scores.max(axis=0)
  ndcg_statistics[2,:] = ndcg_scores.max(axis=0)
  
  for param_idx, params in enumerate(paramGrid):
    cv_scores.append([params, map_statistics[:,param_idx], ndcg_statistics[:,param_idx]])
    
  return cv_scores
##############################################################################
def form_param_grid(ranks, lambdas, alphas):
  paramGrid = list()
  for r in ranks:
    for l in lambdas:
      for a in alphas:
          paramGrid.append([r, l, a])
  
  return paramGrid
          
##############################################################################
def form_and_rdd(train_DF, test_DF, virtual=False):
  train_rdd, test_set = fn.form_tuples(train_DF, test_DF, virtual=virtual)
  train_rdd = sc.parallelize(train_rdd)
  return train_rdd, test_set

##############################################################################
def cv_best_params(cv_scores):
  cv_scores = np.asarray(cv_scores)
  cv_scores = sorted(cv_scores, key=lambda x: x[1,0])
  best_result = cv_scores[-1]
  best_params = best_result[0,:]
  return best_params, best_result, cv_scores

##############################################################################

def before_factorization(train_rdd, test_set, params, n=20):
  print("Starting training...")
  model = train(train_rdd, params)
  metrics = evaluate(model, test_set, n=n)
  map_= metrics.meanAveragePrecision
  precision_= metrics.precisionAt(10)
  ndcg_ = metrics.ndcgAt(10)
  
  return map_, precision_, ndcg_
    
virtual_training, virtual_test = load_user_groups(song_dict, group_size=12)

start_time = time()

print("Initializing Spark....")
spark = SparkSession\
.builder\
.master("local[*]") \
.appName("main")\
.getOrCreate()

sc = spark.sparkContext

paramGrid = form_param_grid([50], [0.01, 0.1, 1.0, 10.0], [0.01, 0.1, 1.0, 10.0, 100.0])

train_rdd, test_set = form_and_rdd(virtual_training, virtual_test, virtual=True)
#scores = list()
#for params in paramGrid:
#  map_, precision_, ndcg_ = before_factorization(train_rdd, test_set, params, n=20)
#  scores.append([params, map_, precision_, ndcg_])
#scores = sorted(scores, key=lambda x: x[1])

map_, precision_, ndcg_ = before_factorization(train_rdd, test_set, [50, 0.1, 0.01], n=200)

#print("Starting cross validation...")
#cv_scores = cross_validation(subsets, paramGrid, n=50)
#best_params, best_result, cv_scores_sorted = cv_best_params(cv_scores)

#train_rdd, test_set = form_and_rdd(train_DF, test_DF)
#model = train(train_rdd, [50, 5.0, 1.0])
#metrics = evaluate(model, test_set, n=200)
#map_= metrics.meanAveragePrecision
#precision_= metrics.precisionAt(10)
#ndcg_= metrics.ndcgAt(10)

elapsed_time = time()-start_time

print("Stopping spark session...")
spark.stop()
print("Stopped.")
  