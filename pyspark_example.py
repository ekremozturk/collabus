#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 23:31:49 2018

@author: seha
"""

from pyspark import SparkContext, SparkConf
from pyspark.sql import *


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

spark = SparkSession\
    .builder\
    .master("local[*]") \
    .appName("Histogram")\
    .getOrCreate()

schema = StructType([ \
    StructField("userID", StringType(), True), \
    StructField("songID", StringType(), True), \
    StructField("count", IntegerType(), True)])
    
lines = spark.read.csv( csv , header = True, schema=schema , sep='\t')
lines.createOrReplaceTempView("triplets")
all_songs = spark.sql("SELECT songID FROM triplets GROUP BY songID")

result = all_songs.collect()

spark.stop()
