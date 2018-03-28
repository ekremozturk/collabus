package com.collabus.taste.spark

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import java.io._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import scala.math.sqrt
import scala.collection.mutable.ListBuffer

object NormalizeSong {
  
  case class Triplet(userID: String, songID: String, playCount: Int)
  
  case class FourTuple(songID: String, playCount: Int, occurCount: Int, mean:Double)
  
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
      
    val pathUN = "/Users/ekrem/No-cloud/datasets4senior/echonest/subset/train_triplets_echonest.txt"
    
    val pathSPM = "/Users/ekrem/No-cloud/datasets4senior/echonest/subset/echonest_song_play_mean.txt"  
      
    val destination = "/Users/ekrem/No-cloud/datasets4senior/echonest/subset/echonest_ii_normalized_subset.txt"
    
    val unLines = spark.sparkContext.textFile(pathUN)

    val unTriplets = unLines.map(parseTriplet)
    
    val spmLines = spark.sparkContext.textFile(pathSPM)
    
    val spmTriplets = spmLines.map(parseFourTuple)
    
    import spark.implicits._
    val schemaUN = unTriplets.toDS
    schemaUN.createOrReplaceTempView("unTriplets")
    
    import spark.implicits._
    val schemaSPM = spmTriplets.toDS
    schemaSPM.createOrReplaceTempView("spmTriplets")
    
    val songMean = spark.sql("SELECT songID, mean FROM spmTriplets")
    
    val triplets = spark.sql("SELECT * FROM unTriplets")
    
    val joined = triplets.join(songMean, "songID")
    
    val nTriplets = joined.withColumn("playCount", $"playCount" - $"mean").drop("mean")
    
    val nTripletsLiked = nTriplets.filter(nTriplets("playCount") > 0.5)
    
    val nTripletsDisliked = nTriplets.filter(nTriplets("playCount") < -0.5)
    
    val nTripletsNatural = nTriplets.filter((nTriplets("playCount") < 0.5) && (nTriplets("playCount") > -0.5))
    
    val result = nTriplets.collect()
    
    writeToFile(result, destination)
    
    spark.stop()

  }
  
  def writeToFile(result: Array[Row], path: String) = {
    
    val pw = new PrintWriter(new File(path))
    
    result.foreach(row=>pw.write(row.get(0).toString()+" "
        +row.get(1).toString()+" "
        +row.get(2).toString()+"\n"))
    
    pw.close
  }
  
  def numOfDistinct(df: DataFrame, field: String): Long = {
    return df.select(field).distinct().count()
  }
  
}