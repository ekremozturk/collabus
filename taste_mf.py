#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:07:13 2018

@author: ekrem
    """

from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from time import time
import pandas as pd
import numpy as np

from pyspark.mllib.recommendation import ALS

import taste_fn as fn

triplets, users, songs = fn.load_files()

userIDs, songIDs = fn.ids(users, songs)

user_dict, song_dict = fn.form_dictionaries(userIDs, songIDs)

print("Replacing IDs with indexes....")
triplets= fn.replace_DF(triplets, user_dict, song_dict)

#print("Splitting into sets....")
#train_DF, test_DF = fn.split_into_train_test(triplets, frac=0.5)

#print("Forming user groups....")
#train_groups, test_groups = fn.load_groups(4)
#virtual_training = fn.form_virtual_users(train_groups, song_dict, agg='normalized_avg')
#virtual_test = fn.form_virtual_users(test_groups, song_dict, agg='normalized_avg')

print("Splitting into subsets....")
subsets = fn.split_into_train_test_cv(triplets, cv=3)

##############################################################################

#def get_records(train_DF, test_DF):
#
#  print("Creating records...")
#
#  record_train = fn.form_records(train_DF, user_dict, song_dict)
#
#  record_test = fn.form_records(test_DF, user_dict, song_dict)
#
#  return record_train, record_test

############################################################################
def evaluate (train_rdd, test_set, params, n=20):
  
  print('Training...')
  rank_, lambda_, alpha_ = params[0], params[1], params[2]
  model = ALS.trainImplicit(train_rdd, 
                            rank_, 
                            iterations=10, 
                            lambda_=lambda_, 
                            alpha=alpha_)

  print("Making recommendations...")
  recommendations = model.recommendProductsForUsers(n).collect()
  
  print("Preparing for metrics...")
  pred_label = fn.prepare_prediction_label(recommendations,test_set)
  prediction_and_labels = sc.parallelize(pred_label)
  metrics = RankingMetrics(prediction_and_labels)

  #return map_, ndcg_
  return metrics
  
############################################################################    
#def cross_validation(subsets, paramGrid):
#  cv_scores = list()
#  for params in paramGrid:
#    
#    map_list =list()
#    ndcg_list = list()
#    for num in range (len(subsets)):
#        
#      print("Creating train sets and test set for trial ", num+1)
#      
#      test_DF = subsets[num]
#      train_DF = pd.concat([element for i, element in enumerate(subsets) if i not in {num}])
#          
#      train_rdd, test_set = form_and_rdd(train_DF, test_DF)
#      
#      metrics = evaluate(train_rdd, test_set, params)
#      map_, ndcg_ = metrics.meanAveragePrecision, metrics.precisionAt(10)
#      map_list.append(map_)
#      ndcg_list.append(ndcg_)
#      
#      avg_ = sum(map_list)/len(map_list)
#      min_, max_ = min(map_list), max(map_list)
#      cv_scores.append([params, map_list, avg_, min_, max_])
#    
#    return cv_scores
#        
############################################################################  
def cross_validation(subsets, paramGrid, n=20):
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
      metrics = evaluate(train_rdd, test_set, params, n)
      map_, ndcg_ = metrics.meanAveragePrecision, metrics.precisionAt(10)
      map_scores[num, param_idx] = map_
      ndcg_scores[num, param_idx] = ndcg_
  
  map_statistics[0,:] = map_scores.mean(axis=0)
  ndcg_statistics[0,:] = ndcg_scores.mean(axis=0)
  map_statistics[1,:] = map_scores.min(axis=0)
  ndcg_statistics[1,:] = ndcg_scores.min(axis=0)
  map_statistics[2,:] = map_scores.max(axis=0)
  ndcg_statistics[2,:] = ndcg_scores.max(axis=0)
  
  for param_idx, params in enumerate(paramGrid):
    cv_scores.append(params, map_statistics[:,param_idx], ndcg_statistics[:,param_idx])
    
  return cv_scores
############################################################################
def form_param_grid(ranks, lambdas, alphas):
  paramGrid = list()
  for r in ranks:
    for l in lambdas:
      for a in alphas:
          paramGrid.append([r, l, a])
  
  return paramGrid
          
############################################################################
def form_and_rdd(train_DF, test_DF, virtual=False):
  train_rdd, test_set = fn.form_tuples(train_DF, test_DF, virtual=virtual)
  train_rdd = sc.parallelize(train_rdd)
  return train_rdd, test_set

############################################################################
start_time = time()

print("Initializing Spark....")
spark = SparkSession\
.builder\
.master("local[*]") \
.appName("main")\
.getOrCreate()

sc = spark.sparkContext

print("Starting cross validation...")
paramGrid = form_param_grid([20, 50, 100, 200], [0.01, 1.0, 10.0], [0.1, 10.0, 40.0])
paramGrid.append([50, 5.0, 10.0])
cv_scores20 = cross_validation(subsets, paramGrid, n=20)
cv_scores50 = cross_validation(subsets, paramGrid, n=50)
cv_scores200 = cross_validation(subsets, paramGrid, n=200)

#map_list = []
#for param in paramGrid:
#  metrics = evaluate(virtual_training, virtual_test, param, virtual=True)
#  map_= metrics.meanAveragePrecision
#  map_list.append([param, map_])

#train_rdd, test_set = form_and_rdd(train_DF, test_DF)
#metrics = evaluate(train_rdd, test_set, [200, 1.0, 10.0])
#map_= metrics.meanAveragePrecision
#ndcg_= metrics.precisionAt(10)

elapsed_time = time()-start_time

print("Stopping spark session...")
spark.stop()
print("Stopped.")


#######################
#model.save("subset/als")
#model = ALSModel.load("subset/als")

