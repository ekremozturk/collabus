package com.collabus.taste.spark

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import java.io._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

object MeanPlayOfUser {
  
  case class Triplet(userID: String, songID: String, playCount: Int)
  
  def parseLines(line: String): Triplet = {
    
    val fields = line.split("\\W+") //Split fields with regex

    val triplet:Triplet = Triplet(fields(0), fields(1), fields(2).toInt) //Map into Triplet object
    
    return triplet
    
  }
  
  def main(args: Array[String]) {
  
  Logger.getLogger("org").setLevel(Level.ERROR)
    
    // Use new SparkSession interface in Spark 2.0
    val spark = SparkSession
      .builder
      .appName("SparkSQL")
      .master("local[*]")
      .getOrCreate()
    
    //Read the file
    val source =  "/Users/ekrem/No-cloud/datasets4senior/echonest/subset/train_triplets_echonest.txt"
      
    val destination = "/Users/ekrem/No-cloud/datasets4senior/echonest/subset/echonest_user_play_mean.txt"
    
    val lines = spark.sparkContext.textFile(source)
    
    val triplets = lines.map(parseLines)
    
    import spark.implicits._
    val schemaLines = triplets.toDS
    schemaLines.createOrReplaceTempView("triplets")
    
    val eachUserTotalPlay = spark.sql("SELECT userID, SUM(playCount) as playCount FROM triplets GROUP BY userID")
    
    val eachUserOccur = spark.sql("SELECT userID, COUNT(userID) as occurence FROM triplets GROUP BY userID")
    
    val occurPlayJoin = eachUserTotalPlay.join(eachUserOccur, "userID")
    
    val meanPlayOfEachUser = occurPlayJoin.withColumn("meanPlay", $"playCount" / $"occurence" )
    
    val result = meanPlayOfEachUser.collect()
    
    val pw = new PrintWriter(new File(destination))
    
    result.foreach(row=>pw.write(row.get(0).toString()+" "
        +row.get(1).toString()+" "
        +row.get(2).toString()+" "
        +row.get(3).toString()+"\n"))
    
    pw.close
    
    println(result.size)
    
    spark.stop()
  }
  
}