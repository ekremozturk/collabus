package com.collabus.taste.spark

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import java.io._
import scala.math.min

object OverallStatistics {
  
  def parseLines(line: String) = {
    val fields = line.split("\\W+")
    val userID = fields(0)
    val songID = fields(1)
    val playCount = fields(2).toInt
    
    (userID, songID, playCount)
    
  }
  
  def main(args: Array[String]) {
    
    Logger.getLogger("org").setLevel(Level.ERROR)
    
    val sc = new SparkContext("local[*]" , "FilterLowPlayCount")
    
    //Read the file
    val lines = sc.textFile("/Users/ekrem/No-cloud/datasets4senior/train_triplets_echonest.txt")
    
    //Parse and create tuples
    val parsedLines = lines.map(parseLines)
    
    //Filter play count 1
    val filterOnePlay = parsedLines.filter(x => x._3 > 1)
    
    //Users whose total play count is more than 100 
    val frequentUsers = filterOnePlay.map(x => (x._1, x._3)).reduceByKey((x,y) => x+y).filter(x => x._2>100)
    
    //Songs which total play count is more than 100 
    val frequentSongs = filterOnePlay.map(x => (x._2, x._3)).reduceByKey((x,y) => x+y).filter(x => x._2>100)
    
    val top10users = frequentUsers.map(x=>(x._2, x._1)).sortByKey(false).take(10)
    val top10songs = frequentSongs.map(x=>(x._2, x._1)).sortByKey(false).take(10)
    
    val numFreqUsers = frequentUsers.count()
    val numFreqSongs = frequentSongs.count()
    
    println("Total number of frequent users: "+ numFreqUsers)
    println("Total number of frequent songs: "+ numFreqSongs)
    
    println("Top 10 users are: ")
    top10users.foreach(println)
    
    println("Top 10 songs are: ")
    top10songs.foreach(println)
    
    //val selectPlayCounts = spark.sql("SELECT playCount FROM triplets")
    
    //val selectUserIDs = spark.sql("SELECT userID FROM triplets GROUP BY userID")
    
    //val selectSongIDs = spark.sql("SELECT songID FROM triplets GROUP BY songID")
    
    //val sumPlayCounts = schemaTriplets.agg(sum("playCount")).first().getLong(0)
    
    //val numOfSongPlay = selectPlayCounts.count()
    
    //val avgPlayCounts = sumPlayCounts.toDouble / numOfSongPlay.toDouble
    
  }
}