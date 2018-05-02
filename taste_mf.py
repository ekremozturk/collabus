#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:07:13 2018

@author: ekrem
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from time import time

import taste_fn as fn

triplets, users, songs = fn.load_files()

train_DF, test_DF = fn.get_subsets()

userIDs, songIDs = fn.ids(users, songs)

user_dict, song_dict = fn.form_dictionaries(userIDs, songIDs)

##############################################################################

def get_records():

  print("Creating records...")

  record_train = fn.form_records(train_DF, user_dict, song_dict)

  record_test = fn.form_records(test_DF, user_dict, song_dict)

  return record_train, record_test

##############################################################################

def to_spark_df(spark, ratings_train, ratings_eval):

  print("Casting to spark dataframe...")

  schema_train = StructType([ \
      StructField("user", IntegerType(), True), \
      StructField("item", IntegerType(), True), \
      StructField("rating", FloatType(), True)])

  ratings_train = spark.createDataFrame(ratings_train, schema_train)

  ratings_eval = spark.createDataFrame(ratings_eval, schema_train)

  return ratings_train, ratings_eval

##############################################################################

def form_tuples(record_train, record_test):

  print("Creating rating tuples...")

  ratings_train = []
  for i in range(record_train.shape[0]):
    for j in range(record_train.shape[1]):
      count = float(record_train[i,j])
      rating = (i, j, count)
      ratings_train.append(rating)

  ratings_test = []
  ratings_eval = []
  for i in range(record_test.shape[0]):
    for j in range(record_test.shape[1]):
      count = record_test[i,j]
      if(count>0):
        rating = (i, j)
        evali = (i, j, 1.0)
        ratings_test.append(rating)
        ratings_eval.append(evali)

  return ratings_train, ratings_eval;

##############################################################################

def tune(ratings_train):

  print("Tuning...")

  ALSmodel = ALS(implicitPrefs=True)

  parameterGrid = ParamGridBuilder() \
    .addGrid(ALSmodel.rank, [20, 50, 100, 200]) \
    .addGrid(ALSmodel.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(ALSmodel.alpha, [10.0, 40.0, 8000.0]) \
    .build()

  evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating")

  cv = CrossValidator(estimator=ALSmodel,
                      estimatorParamMaps=parameterGrid,
                      evaluator=evaluator)

  return  cv.fit(ratings_train)

##############################################################################

def train(data, rank=50, maxIter=10, regParam=0.01, implicitPrefs=True, alpha=40.0):

  print("Training the model...")

  ALSmodel = ALS(rank=rank,
              maxIter=maxIter,
              regParam=regParam,
              implicitPrefs=implicitPrefs,
              alpha=alpha)

  model = ALSmodel.fit(data)

  return ALSmodel, model

##############################################################################
start_time = time()

record_train, record_test = get_records()

ratings_train, ratings_eval = form_tuples(record_train, record_test)

spark = SparkSession\
    .builder\
    .master("local[*]") \
    .appName("main")\
    .getOrCreate()

sc = spark.sparkContext

ratings_train, ratings_eval = to_spark_df(spark, ratings_train, ratings_eval)

#tuning_model = tune(ratings_train)

ALSmodel, model = train(ratings_train)
#model.save("subset/als")

#model = ALSModel.load("subset/als")

predictions = model.transform(ratings_eval)

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating" ,predictionCol="prediction")

rmse = evaluator.evaluate(predictions)

recommendations = model.recommendForAllUsers(20).collect()

print("Stopping spark session...")

spark.stop()

elapsed_time = time()-start_time
print("Stopped.")

#Recommendation'ları userID, songIDs olacak şekilde daha güzel bir hale getir
#Üstteki tamamlanınca ranking metrics implementation