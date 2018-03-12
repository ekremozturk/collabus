package com.collabus.taste.spark

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import java.io._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

object FilterLowPlayCountDS {
  
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
    val lines = spark.sparkContext.textFile("/Users/ekrem/No-cloud/datasets4senior/train_triplets_echonest.txt")
    
    //Map the lines
    val triplets = lines.map(parseLines)

    // Infer the schema, and register the DataSet as a table.
    import spark.implicits._
    val schemaTriplets = triplets.toDS
    schemaTriplets.createOrReplaceTempView("triplets")
    
    schemaTriplets.printSchema()
    
    val selectPlayCounts = spark.sql("SELECT playCount FROM triplets")
    
    val selectUserIDs = spark.sql("SELECT userID FROM triplets GROUP BY userID")
    
    val selectSongIDs = spark.sql("SELECT songID FROM triplets GROUP BY songID")
    
    //val sumPlayCounts = schemaTriplets.agg(sum("playCount")).first().getLong(0)
    
    //val numOfSongPlay = selectPlayCounts.count()
    
    //val avgPlayCounts = sumPlayCounts.toDouble / numOfSongPlay.toDouble
    
    //val scaledPlayCounts = selectPlayCounts.map(x => x.getInt(0) / avgPlayCounts)
    
    //println(scaledPlayCounts.show())
    /**
    val resultPlay = selectPlayCounts.collect()
    
    val pw = new PrintWriter(new File("/Users/ekrem/No-cloud/datasets4senior/echonest_only_playcounts.txt"))
    
    resultPlay.foreach(row=>pw.write(row.get(0).toString()+"\n"))
    
    pw.close
    */
    /**
    val resultUser = selectUserIDs.collect()
    
    val pw2 = new PrintWriter(new File("/Users/ekrem/No-cloud/datasets4senior/echonest_only_userids.txt"))
    
    resultUser.foreach(row=>pw2.write(row.get(0).toString()+"\n"))
    
    pw2.close
    */
    
    val resultSong = selectSongIDs.collect()
    
    val pw3 = new PrintWriter(new File("/Users/ekrem/No-cloud/datasets4senior/echonest_only_songids.txt"))
    
    resultSong.foreach(row=>pw3.write(row.get(0).toString()+"\n"))
    
    pw3.close
 		
    spark.stop()
  }
}