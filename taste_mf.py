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

#=============================================================================
#print("Splitting into sets....")
#train_DF, test_DF = fn.split_into_train_test(triplets, frac=0.5)

#=============================================================================
#print("Splitting into subsets....")
#subsets = fn.split_into_train_test_cv(triplets, cv=2)

#=============================================================================
def load_user_groups(song_dict, group_size=4, agg = 'normalized_avg'):
  print("Forming user groups....")
  train_groups, test_groups = fn.load_groups(group_size)
  virtual_training = fn.form_virtual_users(train_groups, song_dict, agg=agg)
  virtual_test = fn.form_virtual_users(test_groups, song_dict, agg=agg)
  
  return virtual_training, virtual_test

#=============================================================================
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

#=============================================================================
def evaluate(model, test_set, n=20):
  print("Making recommendations...")
  recommendations = model.recommendProductsForUsers(n).collect()
  
  print("Preparing for metrics...", n)
  pred_label = fn.prepare_prediction_label(recommendations,test_set)
  f1_, precision_, recall_ = fn.f1_precision_recall(pred_label)
  mpr_ = fn.mpr(pred_label)
  prediction_and_labels = sc.parallelize(pred_label)
  metrics = RankingMetrics(prediction_and_labels)
  map_ = metrics.meanAveragePrecision
  ndcg_ = metrics.ndcgAt(10)
  return map_, ndcg_, f1_, precision_, recall_, mpr_ 

#============================================================================= 
def cross_validation(subsets, paramGrid, n=200):
  cv_scores = list()
  map_scores = np.zeros((len(subsets), len(paramGrid)))
  mpr_scores = np.zeros((len(subsets), len(paramGrid)))
  map_statistics = np.zeros((3, len(paramGrid)))
  mpr_statistics = np.zeros((3, len(paramGrid)))
  
  for num in range (len(subsets)):
    print("Creating train sets and test set for trial ", num+1)
    
    test_DF = subsets[num]
    train_DF = pd.concat([element for i, element in enumerate(subsets) if i not in {num}])
    train_rdd, test_set = form_and_rdd(train_DF, test_DF)
    
    for param_idx, params in enumerate(paramGrid):
      model = train(train_rdd, params)
      map_, ndcg_, f1_, precision_, recall_, mpr_ = evaluate(model, test_set, n)
      map_scores[num, param_idx] = map_
      mpr_scores[num, param_idx] = mpr_
      print("Completed: " + str(num*len(paramGrid)+param_idx))
  
  map_statistics[0,:] = map_scores.mean(axis=0)
  mpr_statistics[0,:] = mpr_scores.mean(axis=0)
  map_statistics[1,:] = map_scores.min(axis=0)
  mpr_statistics[1,:] = mpr_scores.min(axis=0)
  map_statistics[2,:] = map_scores.max(axis=0)
  mpr_statistics[2,:] = mpr_scores.max(axis=0)
  
  for param_idx, params in enumerate(paramGrid):
    cv_scores.append([params, map_statistics[:,param_idx], mpr_statistics[:,param_idx]])
    
  return cv_scores
#=============================================================================
def form_param_grid(ranks, lambdas, alphas):
  paramGrid = list()
  for r in ranks:
    for l in lambdas:
      for a in alphas:
          paramGrid.append([r, l, a])
  
  return paramGrid
          
#=============================================================================
def form_and_rdd(train_DF, test_DF, virtual=False):
  train_rdd, test_set = fn.form_tuples(train_DF, test_DF, virtual=virtual)
  train_rdd = sc.parallelize(train_rdd)
  return train_rdd, test_set

#=============================================================================
def cv_best_params(cv_scores):
  cv_scores = np.asarray(cv_scores)
  cv_scores = sorted(cv_scores, key=lambda x: x[2,0])
  best_result = cv_scores[0]
  best_params = best_result[0,:]
  return best_params, best_result, cv_scores

#=============================================================================

def before_factorization(train_rdd, test_set, params, n=20):
  print("Starting training...")
  model = train(train_rdd, params)  
  return evaluate(model, test_set, n=n)
    
#=============================================================================
virtual_training, virtual_test = load_user_groups(song_dict, group_size=12)

#paramGrid = form_param_grid([100], [0.01, 0.1, 1.0, 10.0], [0.01, 0.1, 1.0, 10.0])
#paramGrid = form_param_grid([50, 100, 200], [0.1], [0.1])

start_time = time()
print("Initializing Spark....")
spark = SparkSession\
.builder\
.master("local[*]") \
.appName("main")\
.getOrCreate()

sc = spark.sparkContext

#train_rdd, test_set = form_and_rdd(train_DF, test_DF)
train_rdd, test_set = form_and_rdd(virtual_training, virtual_test, virtual=True)

#scores = list()
#for params in paramGrid:
#  map_, ndcg_, f1_, precision_, recall_, mpr_ = before_factorization(train_rdd, test_set, params, n=200)
#  scores.append([params, map_, ndcg_, f1_, precision_, recall_, mpr_])
#scores = sorted(scores, key=lambda x: x[1])

#map_, ndcg_, f1_, precision_, recall_, mpr_ = before_factorization(train_rdd, test_set, [20, 0.01, 0.01], n=20)

##CV PARAMETER TUNING
#print("Starting cross validation...")
#cv_scores = cross_validation(subsets, paramGrid, n=50)
#best_params, best_result, cv_scores_sorted = cv_best_params(cv_scores)

scores = list()
model = train(train_rdd, [10, 0.1, 0.01])
for n_ in fn.n_list:
  scores.append(evaluate(model, test_set, n=n_))
scores_mf = pd.DataFrame(scores, index = [fn.n_list], columns=['mAP', 'NDCG', 'F1', 'Precision', 'Recall', 'mPR'])

#map_, ndcg_, f1_, precision_, recall_, mpr_ = evaluate(model, test_set, n=200)
  
elapsed_time = time()-start_time

print("Stopping spark session...")
spark.stop()
print("Stopped.")

#triplets, users, songs = fn.load_files()
#print("Splitting into sets....")
#train_DF, test_DF = fn.split_into_train_test(triplets, frac=0.5)
#fn.form_and_save_groups(userIDs, train_DF, test_DF)