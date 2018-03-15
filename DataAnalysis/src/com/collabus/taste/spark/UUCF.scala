package com.collabus.taste.spark

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import java.io._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import scala.math.sqrt
import scala.collection.mutable.ListBuffer

object UUCF {
  
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
    
    val spark = SparkSession
      .builder
      .appName("SparkSQL")
      .master("local[*]")
      .getOrCreate()
      
    val pathUN = "/Users/ekrem/No-cloud/datasets4senior/train_triplets_echonest.txt"
    
    val pathUPM = "/Users/ekrem/No-cloud/datasets4senior/echonest_user_play_mean.txt"  
      
    val unLines = spark.sparkContext.textFile(pathUN)

    val unTriplets = unLines.map(parseTriplet)
    
    val upmLines = spark.sparkContext.textFile(pathUPM)
    
    val upmTriplets = upmLines.map(parseFourTuple)
    
    import spark.implicits._
    val schemaUN = unTriplets.toDS
    schemaUN.createOrReplaceTempView("unTriplets")
    
    import spark.implicits._
    val schemaUPM = upmTriplets.toDS
    schemaUPM.createOrReplaceTempView("upmTriplets")
    
    val userMean = spark.sql("SELECT userID, mean FROM upmTriplets")
    
    val triplets = spark.sql("SELECT * FROM unTriplets")
    
    val I = 5 //common song subset min limit
    
    val k = 30 //k-NN
    
    val joined = triplets.join(userMean, "userID")
    
    val nTriplets = joined.withColumn("playCount", $"playCount" - $"mean").drop("mean")
    
    val nTripletsOnlyLiked = nTriplets.filter(nTriplets("playCount") > 0)
    
    println(nTriplets.show)

    spark.stop()

  }
  
  def pearsonCorr(r_u: Array[Row], r_v: Array[Row]): Double = {
    
    //Assuming the play counts are normalized
    
    var cov = 0.0
    
    var var_u = 0.0
    
    var var_v = 0.0
    
    r_u.foreach(row_u => r_v.foreach(row_v => cov += row_u.getDouble(0)*row_v.getDouble(0)))
    
    r_u.foreach(row_u => var_u += row_u.getDouble(0)*row_u.getDouble(0))
    
    r_v.foreach(row_v => var_v += row_v.getDouble(0)*row_v.getDouble(0))
    
    return cov/(sqrt(var_u)*sqrt(var_v))
  }
  
  def numOfDistinct(df: DataFrame, field: String): Long = {
    return df.select(field).distinct().count()
  }
  
}