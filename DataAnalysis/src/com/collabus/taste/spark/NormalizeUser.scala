package com.collabus.taste.spark

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import java.io._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import scala.math.sqrt
import scala.collection.mutable.ListBuffer

object NormalizeUser {
  
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
    
    //val nTripletsLiked = nTriplets.filter(nTriplets("playCount") > 0.5)
    
    //val nTripletsDisliked = nTriplets.filter(nTriplets("playCount") < -0.5)
    
    //val nTripletsNatural = nTriplets.filter((nTriplets("playCount") < 0.5) && (nTriplets("playCount") > -0.5))

    val randUser = userMean.select("userID").sample(false, 0.01)
    
    val randTriplets = nTriplets.join(randUser, "userID").cache()
    
    val firstUser = randUser.first()
    
    val triplet_u = randTriplets.filter(randTriplets("userID") === firstUser.getString(0))
 
    println(triplet_u.toString())
    
    val resultRandUser = randUser.collect()
    
    for(user <- resultRandUser){
      
      val currentID = user.getString(0)
      
      val triplet_v = randTriplets.filter(randTriplets("userID") === currentID) 
      
      val joined = triplet_u.join(triplet_v, "songID").drop("songID", "userID").toDF("U", "V").cache()
      
      if(joined.count()>0){
        val r_u = joined.select("U").collect()
        val r_v = joined.select("V").collect()
        println((currentID + " " + pearsonCorr(r_u, r_v)))
      }
    }
    
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