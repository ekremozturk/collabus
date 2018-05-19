package com.collabus.taste.spark

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import java.io._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

object ExtractSubset {
  
  case class Triplet(userID: String, songID: String, playCount: Int)
  
  case class FourTuple(ID: String, playCount: Int, occurCount: Int, mean:Double)
  
  def parseTriplet(line: String): Triplet = {
    
    val fields = line.split("\\W+") //Split fields with regex

    val triplet:Triplet = Triplet(fields(0), fields(1), fields(2).toInt) //Map into Triplet object
    
    return triplet
    
  }
  
  def parseFourTuple(line: String): FourTuple = {
    
    val fields = line.split(" ") //Split fields with regex

    val fourTuple:FourTuple = FourTuple(fields(0), fields(1).toInt, fields(2).toInt, fields(3).toDouble)
    
    return fourTuple
    
  }
  
  def main(args: Array[String]) {
  
    Logger.getLogger("org").setLevel(Level.ERROR)
    
    val spark = SparkSession
      .builder
      .appName("SparkSQL")
      .master("local[*]")
      .getOrCreate()
      
    val pathUN = "/Users/ekrem/No-cloud/datasets4senior/echonest/train_triplets_echonest.txt"
      
    val unLines = spark.sparkContext.textFile(pathUN)

    val unTriplets = unLines.map(parseTriplet)
    
    import spark.implicits._
    val schemaUN = unTriplets.toDS
    schemaUN.createOrReplaceTempView("unTriplets")
    
    val pathSPM = "/Users/ekrem/No-cloud/datasets4senior/echonest/echonest_song_play_mean.txt"
    
    val spmLines = spark.sparkContext.textFile(pathSPM)
    
    val spmTriplets = spmLines.map(parseFourTuple)
    
    import spark.implicits._
    val schemaSPM = spmTriplets.toDS
    schemaSPM.createOrReplaceTempView("spmTriplets")
    
    val triplets = spark.sql("SELECT * FROM unTriplets")
    
    val songSubset = spark.sql("SELECT ID FROM spmTriplets").sample(true, 0.03)
    
    //val joinUser = triplets.join(userSubset, $"userID"===$"ID").drop("ID")
    
    val joinSong = triplets.join(songSubset, $"songID"===$"ID").drop("ID").cache()
    
    val valUsers = joinSong.groupBy("userID").count().filter($"count">7).drop("count")
    
    val subset = joinSong.join(valUsers, "userID").cache()
    
    val numDistinctUsers = numOfDistinct(subset, "userID")
    
    val numDistinctSongs = numOfDistinct(subset, "songID")
    
    val result = subset.collect()
    //val result = subset.count()
    writeToFile(result, "/Users/ekrem/No-cloud/datasets4senior/echonest/subset/train_triplets.txt")
    
    println("There are "+numDistinctUsers+" users, "+numDistinctSongs+" songs, "+result.length+" triplets")
    //println("There are "+numDistinctUsers+" users, "+numDistinctSongs+" songs, "+result+" triplets")
    spark.stop()
      
  }
  
  def numOfDistinct(df: DataFrame, field: String): Long = {
    return df.select(field).distinct().count()
  }
  
  def writeToFile(result: Array[Row], path: String) = {
    
    val pw = new PrintWriter(new File(path))
    
    result.foreach(row=>pw.write(row.get(0).toString()+" "
        +row.get(1).toString()+" "
        +row.get(2).toString()+"\n"))
    
    pw.close
  }
  
}