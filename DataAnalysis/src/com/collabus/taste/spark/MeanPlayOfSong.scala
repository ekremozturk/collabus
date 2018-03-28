package com.collabus.taste.spark

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import java.io._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

object MeanPlayOfSong {
  
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
    val source = "/Users/ekrem/No-cloud/datasets4senior/echonest/subset/train_triplets_echonest.txt"
    
    val destination = "/Users/ekrem/No-cloud/datasets4senior/echonest/subset/echonest_song_play_mean.txt"
      
    val lines = spark.sparkContext.textFile(source)
    
    //Map the lines
    val triplets = lines.map(parseLines)

    // Infer the schema, and register the DataSet as a table.
    import spark.implicits._
    val schemaTriplets = triplets.toDS
    schemaTriplets.createOrReplaceTempView("triplets")
    
    val playCountForEachSong = spark.sql("SELECT songID, SUM(playCount) as playCount FROM triplets GROUP BY songID")
    
    val occurOfEachSong = spark.sql("SELECT songID, COUNT(songID) as occurence FROM triplets GROUP BY songID")
    
    val occurPlayJoin = playCountForEachSong.join(occurOfEachSong, "songID")
    
    val meanPlayForEachSong = occurPlayJoin.withColumn("meanPlay", $"playCount" / $"occurence" )
    
    val result = meanPlayForEachSong.collect()
    
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