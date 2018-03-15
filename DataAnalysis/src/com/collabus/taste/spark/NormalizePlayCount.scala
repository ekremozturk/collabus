package com.collabus.taste.spark

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import java.io._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

object NormalizePlayCount {
  
  case class Triplet(userID: String, songID: String, playCount: Int)
  
  case class FourTuple(userID: String, playCount: Int, occurCount: Int, mean:Double)
  
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
    
    // Use new SparkSession interface in Spark 2.0
    val spark = SparkSession
      .builder
      .appName("SparkSQL")
      .master("local[*]")
      .getOrCreate()
    
    // un: unnormalized, n: normalized, upm: user play mean  
      
    //Read the file
    val unLines = spark.sparkContext.textFile("/Users/ekrem/No-cloud/datasets4senior/train_triplets_echonest.txt")
    
    val upmLines = spark.sparkContext.textFile("/Users/ekrem/No-cloud/datasets4senior/echonest_user_play_mean.txt")
    
    val unTriplets = unLines.map(parseTriplet)
    
    val upmTriplets = upmLines.map(parseFourTuple)
    
    import spark.implicits._
    val schemaUN = unTriplets.toDS
    schemaUN.createOrReplaceTempView("unTriplets")
    
    import spark.implicits._
    val schemaUPM = upmTriplets.toDS
    schemaUPM.createOrReplaceTempView("upmTriplets")
    
    val userMean = spark.sql("SELECT userID, mean FROM upmTriplets")
    
    val triplets = spark.sql("SELECT * FROM unTriplets")
    
    val joined = triplets.join(userMean, "userID")
    
    val nTriplets = joined.withColumn("playCount", $"playCount" - $"mean")
    
    println(nTriplets.show(200))
    
    spark.stop()
  }
  
}