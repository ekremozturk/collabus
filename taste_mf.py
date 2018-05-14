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

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

import taste_fn as fn
        
#train_rdd, test_set = fn.form_tuples(virtual_training, virtual_test, virtual=True)

start_time = time()

triplets, users, songs = fn.load_files()

userIDs, songIDs = fn.ids(users, songs)

user_dict, song_dict = fn.form_dictionaries(userIDs, songIDs)

print("Replacing IDs with indexes....")
triplets= fn.replace_DF(triplets, user_dict, song_dict)

print("Splitting into sets....")
train_DF, test_DF = fn.split_into_train_test(triplets)
#print("Forming user groups....")
#userGroups = fn.group_users(userIDs ,12)
#train_groups, test_groups = fn.form_groups(userGroups, train_DF, test_DF)
#virtual_training = fn.form_virtual_users(train_groups, song_dict, agg='average')
#virtual_test = fn.form_virtual_users(test_groups, song_dict, agg='average')

#print("Splitting into subsets....")
#subsets = fn.split_into_train_test_cv(triplets)

##############################################################################

def get_records(train_DF, test_DF):

  print("Creating records...")

  record_train = fn.form_records(train_DF, user_dict, song_dict)

  record_test = fn.form_records(test_DF, user_dict, song_dict)

  return record_train, record_test

############################################################################
def evaluate (train_DF, test_DF, params):
  
  train_rdd, test_set = fn.form_tuples(train_DF, test_DF)
  
  train_rdd = sc.parallelize(train_rdd)
  
  rank_, lambda_, alpha_ = params[0], params[1], params[2]
  model = ALS.trainImplicit(train_rdd, 
                            rank_, 
                            iterations=10, 
                            lambda_=lambda_, 
                            alpha=alpha_)

  print("Making recommendations...")
  recommendations = model.recommendProductsForUsers(50).collect()
  
  print("Preparing for metrics...")
  pred_label = fn.prepare_prediction_label(recommendations,test_set)
  prediction_and_labels = sc.parallelize(pred_label)
  metrics = RankingMetrics(prediction_and_labels)
  map_= metrics.meanAveragePrecision
  ndcg_= metrics.precisionAt(20)

  return map_, ndcg_
  
############################################################################    
def cross_validation(subsets, paramGrid):
  cv_scores = list()
  for params in paramGrid:
    
    map_list =list()
    ndcg_list = list()
    for num in range (len(subsets)):
        
      print("Creating train sets and test set for trial ", num+1)
      
      test_DF = subsets[num]
      train_DF = pd.concat([element for i, element in enumerate(subsets) if i not in {num}])
          
      map_, ndcg_ = evaluate(train_DF, test_DF, params)
      map_list.append(map_)
      ndcg_list.append(ndcg_)
      
      avg_ = sum(map_list)/len(map_list)
      min_, max_ = min(map_list), max(map_list)
      cv_scores.append([params, map_list, avg_, min_, max_])
    
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

print("Initializing Spark....")
spark = SparkSession\
.builder\
.master("local[*]") \
.appName("main")\
.getOrCreate()

sc = spark.sparkContext

print("Starting cross validation...")
paramGrid = form_param_grid([20, 50], [1.0, 3.5, 7.0], [10.0, 20.0])

#cv_scores = cross_validation(subsets, paramGrid)

#map_, ndcg_ = evaluate(virtual_training, virtual_test, [20, 1.0, 10.0])
map_, ndcg_ = evaluate(train_DF, test_DF, [20, 1.0, 10.0])

print("Stopping spark session...")
spark.stop()
print("Stopped.")
elapsed_time = time()-start_time

#######################3
#model.save("subset/als")
#model = ALSModel.load("subset/als")

