#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:07:13 2018

@author: ekrem
    """

from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics
from time import time

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

import taste_fn as fn

start_time = time()

triplets, users, songs = fn.load_files()

userIDs, songIDs = fn.ids(users, songs)

user_dict, song_dict = fn.form_dictionaries(userIDs, songIDs)

#lis = fn.split_into_train_test(triplets)
subsets = fn.split_into_train_test(triplets)

test_DF = subsets[3]
train_DF = pd.concat([element for i, element in enumerate(subsets) if i not in {3}])

#train_DF, test_DF = fn.split_into_train_test(triplets)


#train_DF, test_DF = fn.replace_DF(train_DF, test_DF, user_dict, song_dict)

#userGroups = fn.group_users(userIDs ,12)

#train_groups, test_groups = fn.form_groups(userGroups, train_DF, test_DF)

#virtual_training = fn.form_virtual_users(train_groups, song_dict, agg='average')

#virtual_test = fn.form_virtual_users(test_groups, song_dict, agg='average')

##############################################################################

def get_records():

  print("Creating records...")

  record_train = fn.form_records(train_DF, user_dict, song_dict)

  record_test = fn.form_records(test_DF, user_dict, song_dict)

  return record_train, record_test


#train_rdd = []
#for idx, row in train_DF.iterrows():
#  train_rdd.append((row[0], row[1], row[2]))
#
#test_set = []
#for idx, row in test_DF.iterrows():
#  test_set.append((row[0], row[1]))
  
record_train, record_test = get_records()

train_rdd, test_set = fn.form_tuples(record_train, record_test)

#train_rdd, test_set = fn.form_tuples(virtual_training, virtual_test, virtual=True)

spark = SparkSession\
    .builder\
    .master("local[*]") \
    .appName("main")\
    .getOrCreate()

sc = spark.sparkContext

train_rdd = sc.parallelize(train_rdd)

model = ALS.trainImplicit(train_rdd, 50, iterations=10, lambda_=1.1, alpha=10.0)

print("Making recommendations...")
recommendations = model.recommendProductsForUsers(50).collect()

print("Preparing for metrics...")
pred_label = fn.prepare_prediction_label(recommendations,test_set)
prediction_and_labels = sc.parallelize(pred_label)
metrics = RankingMetrics(prediction_and_labels)
mean_avg_precision= metrics.meanAveragePrecision
ndcg= metrics.precisionAt(20)

elapsed_time = time()-start_time

print("Stopping spark session...")
spark.stop()
print("Stopped.")

#model.save("subset/als")
#model = ALSModel.load("subset/als")

