#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 23:31:49 2018

@author: seha
"""

from pyspark import SparkContext, SparkConf
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *


count_threshold = 10


class Triplet:
    def __init__(self, userID, songID, playCount):
        self.userID = userID
        self.songID = songID
        self.playCount = playCount
        
def parseLines(line):
    fields = line.split()
    triplet = Triplet (fields[0], fields[1], int(fields[2]))        
    return triplet

csv = "/home/seha/workspace/listening_matrix_extractor/train_triplets.csv"
test = "/home/seha/workspace/listening_matrix_extractor/lol.csv"


spark = SparkSession\
    .builder\
    .master("local[*]") \
    .appName("Histogram")\
    .getOrCreate()

schema = StructType([ \
    StructField("userID", StringType(), True), \
    StructField("songID", StringType(), True), \
    StructField("listen_count", IntegerType(), True)])
    
lines = spark.read.csv( csv , header = True, schema=schema , sep='\t')
lines.createOrReplaceTempView("triplets")

    
df1 = spark.sql("SELECT songID, listen_count FROM triplets ")\
    .filter("listen_count > 10")\
    .groupBy("songID")\
    .count()\
    
df1.createOrReplaceTempView("df1")
df1 = df1.withColumnRenamed('count','liked')


    
df2 = spark.sql("SELECT songID, listen_count FROM triplets ")\
    .filter("listen_count <= 3")\
    .groupBy("songID")\
    .count()\

df2.createOrReplaceTempView("df2")
df2 = df2.withColumnRenamed('count','not_liked')


df3 = df1.join(df2, 'songID', 'outer').select('*')

df3= df3.na.fill(0)
df3 = df3.withColumn('total', df3['liked']+ df3['not_liked'])
df3 = df3.withColumn('liking_percent' , df3['liked'] / df3['total'])

number_of_songs = 308000 #fix this

df3 = df3.withColumn('listen_percent', df3['total']/number_of_songs )

songs = df3.collect()

spark.stop()
