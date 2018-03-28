#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:07:13 2018

@author: ekrem
"""

from pyspark import SparkContext, SparkConf
from pyspark.sql import *

class Triplet:
    def __init__(self, userID, songID, playCount):
        self.userID = userID
        self.songID = songID
        self.playCount = playCount
        
def parseLines(line):
    fields = line.split(" ")
    triplet = Triplet (fields[0], fields[1], int(fields[2]))        
    return triplet

path = "Subset/echonest_song_play_mean.txt"

spark = SparkSession\
    .builder\
    .master("local[*]") \
    .appName("main")\
    .getOrCreate()

schema = StructType([ \
    StructField("userID", StringType(), True), \
    StructField("songID", StringType(), True), \
    StructField("count", IntegerType(), True)])
    

lines.createOrReplaceTempView("triplets")


all_songs = spark.sql("SELECT songID FROM triplets GROUP BY songID")

result = all_songs.collect()

spark.stop()        
