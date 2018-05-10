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
from pyspark.mllib.evaluation import RankingMetrics
from time import time

import taste_fn as fn

start_time = time()

triplets, users, songs = fn.load_files()

train_DF, test_DF = fn.get_subsets()

userIDs, songIDs = fn.ids(users, songs)

user_dict, song_dict = fn.form_dictionaries(userIDs, songIDs)

userGroups = fn.group_users(userIDs ,12)

train_groups, test_groups = fn.form_groups(userGroups, train_DF, test_DF)

virtual_training = fn.form_virtual_users(train_groups, song_dict, agg='normalized_avg')

virtual_test = fn.form_virtual_users(test_groups, song_dict, agg='normalized_avg')

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

  ALES = ALS(rank=rank,
              maxIter=maxIter,
              regParam=regParam,
              implicitPrefs=implicitPrefs,
              alpha=alpha)

  model = ALES.fit(data)

  return ALES, model

##############################################################################

record_train, record_test = get_records()

ratings_train, ratings_eval = fn.form_tuples(record_train, record_test)

#ratings_train, ratings_eval = form_tuples(virtual_training, virtual_test, virtual=True)

spark = SparkSession\
    .builder\
    .master("local[*]") \
    .appName("main")\
    .getOrCreate()

sc = spark.sparkContext

ratings_train, _ = to_spark_df(spark, ratings_train, ratings_eval)

#tuning_model = tune(ratings_train)

als, model = train(ratings_train)
#model.save("subset/als")

#model = ALSModel.load("subset/als")

#predictions = model.transform(ratings_eval)

#evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating" ,predictionCol="prediction")

#rmse = evaluator.evaluate(predictions)

print("Making recommendations...")
recommendations = model.recommendForAllUsers(500).collect()

print("Preparing for metrics...")
pred_label = fn.prepare_prediction_label(recommendations,ratings_eval)
prediction_and_labels = sc.parallelize(pred_label)
metrics = RankingMetrics(prediction_and_labels)
mean_avg_prec= metrics.meanAveragePrecision

print("Stopping spark session...")

spark.stop()

elapsed_time = time()-start_time
print("Stopped.")

#Recommendation'ları userID, songIDs olacak şekilde daha güzel bir hale getir
#Üstteki tamamlanınca ranking metrics implementation
#Group recommendation