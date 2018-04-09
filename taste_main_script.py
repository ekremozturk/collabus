#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:07:13 2018

@author: ekrem
"""

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pyspark.sql.functions import *
from pyspark.mllib.recommendation import *

######################################################
def read_four_tuple(spark, path):
    
  sc = spark.sparkContext

  schema = StructType([ \
      StructField("ID", StringType(), True), \
      StructField("total", IntegerType(), True), \
      StructField("occurence", IntegerType(), True), \
      StructField("mean", FloatType(), True)])
  
  lines = sc.textFile(path)
  parts = lines.map(lambda l: l.split(" "))
  four_tuples = parts.map(lambda p: (p[0], int(p[1]), int(p[2]), float(p[3])))
  
  four_tuple_schema = spark.createDataFrame(four_tuples, schema)
  
  four_tuple_schema.createOrReplaceTempView("four_tuples")
      
  four_tuple_DF = spark.sql("SELECT * FROM four_tuples")
  
  return four_tuple_DF  

######################################################
def read_triplet(spark, path):
  
  sc = spark.sparkContext

  schema = StructType([ \
      StructField("userID", StringType(), True), \
      StructField("songID", StringType(), True), \
      StructField("playCount", IntegerType(), True)])
  
  lines = sc.textFile(path)
  parts = lines.map(lambda l: l.split(" "))
  triplets = parts.map(lambda p: (p[0], p[1], int(p[2])))
  
  triplet_schema = spark.createDataFrame(triplets, schema)
  
  triplet_schema.createOrReplaceTempView("triplets")
      
  triplet_DF = spark.sql("SELECT * FROM triplets")
  
  return triplet_DF

######################################################
def write_triplet(triplets, path):
  file = open(path ,"w") 
  for t in triplets:
    file.write(t[0]+" "+t[1]+" "+str(t[2])+"\n")
  file.close() 
  
######################################################
def split_into_train_test(df, frac):
  train = df.toPandas().groupby("userID", group_keys=False).apply(lambda df: df.sample(frac=0.8))
  test = df.toPandas().drop(train.index)
  return train, test

######################################################
def pandas_to_pyspark(spark, df, schema):
  return spark.createDataFrame(df, schema)

######################################################
def subset_into_train_test(spark, df, train_frac=0.8):

  train, test = split_into_train_test(df, train_frac)

  path_train = "subset/train_triplets.txt"

  path_test = "subset/test_triplets.txt"
  
  schema = StructType([ \
        StructField("userID", StringType(), True), \
        StructField("songID", StringType(), True), \
        StructField("playCount", IntegerType(), True)])
    
  train = pandas_to_pyspark(spark, train, schema).collect() 
  test  = pandas_to_pyspark(spark, test, schema).collect()
  
  write_triplet(train, path_train)
  
  write_triplet(test, path_test)
  
  train_DF = read_triplet(spark, path_train)
  
  test_DF = read_triplet(spark, path_test)
  
  return train_DF, test_DF  

######################################################
  
def get_subsets(spark):
  
  path_train = "subset/train_triplets.txt"

  path_test = "subset/test_triplets.txt"
  
  return read_triplet(spark, path_train), read_triplet(spark, path_test)

######################################################
def record_matrix(train_list, user_dict, song_dict): 
      
  record = np.zeros((len(user_dict), len(song_dict) ), dtype=np.int32)
  
  for t in train_list:
    user_idx = user_dict[t[0]]
    song_idx = song_dict[t[1]]
    listen_count = t[2]
    record[user_idx, song_idx] = listen_count 
    
  return record

######################################################
def pref_matrix(record):
  P = np.zeros((record.shape[0], record.shape[1] ), dtype=np.int32)
  
  for i in range(record.shape[0]):
    for j in range(record.shape[1]):
      if record[i,j]>0:
        P[i,j] = 1
        
  return P

######################################################
def conf_matrix(record):
  C = np.zeros((record.shape[0], record.shape[1] ), dtype=np.int32)
  alpha = 40
  for i in range(record.shape[0]):
    for j in range(record.shape[1]):
      C[i,j] = 1+record[i,j]*alpha
        
  return C

####################################################
def form_dictionaries(users_list, songs_list):
  song_dict = dict()
  user_dict = dict()
  
  count = 0
  for item in songs_list:
      song_dict[item[0]] = count
      count += 1 
  
  count = 0
  for item in users_list:
      user_dict[item[0]] = count
      count += 1
  
  return user_dict, song_dict

####################################################
def train_MF(train_rdd, test_rdd, f=20, a=0.01, l=0.01):
  model = ALS.trainImplicit(train_rdd, f, seed=10, lambda_=l, alpha=a, iterations=10)
  result = model.predictAll(test_rdd)
  return result

####################################################
def evaluate_cm(rap_result):
  cm = np.zeros((2,2), dtype=np.int32)

  for r in rap_result:
    if r[1][0]==0 and r[1][1] == 0:
      cm[0,0] = cm[0,0]+1
    elif r[1][0]==0 and r[1][1] == 1:
      cm[0,1] = cm[0,1]+1
    elif r[1][0]==1 and r[1][1] == 0:
      cm[1,0] = cm[1,0]+1
    else:
      cm[1,1] = cm[1,1]+1
      
  return cm
####################################################
  
def get_records(spark):
  path_triplets = "subset/train_triplets_echonest.txt"

  path_songs = "subset/echonest_song_play_mean.txt"
     
  path_users = "subset/echonest_user_play_mean.txt"
  
  print("Loading triplets...")
  
  triplets_DF = read_triplet(spark, path_triplets)
  
  songs_DF = read_four_tuple(spark, path_songs)
  
  users_DF = read_four_tuple(spark, path_users)
  
  print("Splitting into train and test sets...")
  
  #train_DF, test_DF = subset_into_train_test(spark, triplets_DF)
  
  train_DF, test_DF = get_subsets(spark)
  
  print("Collecting dataframes...")
  
  songs_list = songs_DF.collect()
  users_list = users_DF.collect()
  train_list = train_DF.collect()
  test_list = test_DF.collect()

  print("Creating records...")
  
  user_dict, song_dict = form_dictionaries(users_list, songs_list)
  
  record_train = record_matrix(train_list, user_dict, song_dict)
  
  record_test = record_matrix(test_list, user_dict, song_dict)
  
  return record_train, record_test
####################################################

def result_MF(spark, record_train, record_test, f, a, l):
  print("Creating rating tuples...")

  sc = spark.sparkContext
  
  ratings_train = []
  for i in range(record_train.shape[0]):
    for j in range(record_train.shape[1]):
      count = record_train[i,j]
      rating = (i, j, count)
      ratings_train.append(rating)
        
  ratings_test = []
  ratings_eval = []
  for i in range(record_test.shape[0]):
    for j in range(record_test.shape[1]):
      count = record_test[i,j]
      if(count>0):
        rating = (i, j)
        evali = (i, j, 1)
        ratings_test.append(rating)
        ratings_eval.append(evali)
        
  print("Creating RDDs...")
  
  train_rdd = sc.parallelize(ratings_train)
  test_rdd = sc.parallelize(ratings_test)
  eval_rdd = sc.parallelize(ratings_eval)

  rap_result = train_evaluate_MF( train_rdd, test_rdd, eval_rdd, f, a, l)
  
  return rap_result;
####################################################

def train_evaluate_MF(train_rdd, test_rdd, eval_rdd, f=20, a=0.01, l=0.01):
  print("Training...")
  
  aa = train_MF(train_rdd, test_rdd, f, a, l)
  
  #r_x = np.asarray(aa.map(lambda r: r[2]).collect())
  #r_min = np.min(r_x)
  #r_max = np.max(r_x)
  
  #result = aa.map(lambda r: ((r[0], r[1]), int(((r[2]-r_min)/(r_max-r_min))>0.5)))
  result = aa.map(lambda r: ((r[0], r[1]), r[2]))
  print("Joining predicted and actual results...")
  
  ratesAndPreds = eval_rdd.map(lambda r: ((r[0], r[1]), r[2])).join(result)
  rap_result = ratesAndPreds.collect()
  
  print("Evaluating results...")
  
  MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
  
  print("rank= "+str(f)+", alpha= "+str(a)+", lambda= "+str(l)+", MSE: "+str(MSE))
  
  return [f, a, l, MSE];

####################################################


parameters = {'a': [80.0, 160.0, 320.0], 'l': [0.01, 0.1, 1.0], 'f': [20]}
results = []
for f in parameters['f']:
  for l in parameters['l']:
    for a in parameters['a']:
      spark = SparkSession\
          .builder\
          .master("local[*]") \
          .appName("main")\
          .getOrCreate()  
      sc = spark.sparkContext
      record_train, record_test = get_records(spark)
      results.append(result_MF(spark, record_train, record_test, f, a, l))
      print("Stopping spark session...")  
      spark.stop()
      print("Stopped.") 