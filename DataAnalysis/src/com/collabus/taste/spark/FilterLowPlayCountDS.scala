package com.collabus.taste.spark

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import java.io._
import org.apache.spark.sql._

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
    
    val filteredTriplets = spark.sql("SELECT * FROM triplets WHERE playCount > 2")
    
    //val valuableTriplets = filteredTriplets.select("userID").groupBy("userID").count()
    
    //val result = filteredTriplets.collect()
    
    println(filteredTriplets.count())
    
    val pw = new PrintWriter(new File("/Users/ekrem/No-cloud/datasets4senior/train_triplets_echonest_filtered.txt"))
    
    /*
    for(row <- filteredTriplets){
      
      val userID = row(0).toString()
      val songID = row(1).toString()
      val count = row(2).toString()
      
      pw.write(userID + " " + songID+ "\n")
      
    }
    * 
    */
    pw.close
    
    /**
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
    
    //extract valuable user-song-play_count pairs
    //val valuableUserSongPairs = filteredPlayCounts.filter(x=> )
    
    //collect (rdd action)
    //val result = valuableUserSongPairs.collect()
    
    //val pw = new PrintWriter(new File("/Users/ekrem/No-cloud/datasets4senior/train_triplets_echonest_filtered.txt"))
    
    //result.foreach(pair => pw.write(pair._1 + " " + pair._2 + " " +pair._3 + "\n"))
    
    //pw.close
    //println("Finished "+ frequentUsers.count())
    //println("Finished "+ valuableUserSongPairs.count())
     * 
     */
    spark.stop()
  }
}