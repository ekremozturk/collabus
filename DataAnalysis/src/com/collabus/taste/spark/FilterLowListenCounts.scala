package com.collabus.taste.spark

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import java.io._
import scala.math.min

object FilterLowListenCounts {
  
  def parseLines(line: String) = {
    val fields = line.split("\\W+")
    val userID = fields(0)
    val songID = fields(1)
    val playCount = fields(2).toInt
    
    (userID, songID, playCount)
    
  }
  
  def main(args: Array[String]) {
    
    Logger.getLogger("org").setLevel(Level.ERROR)
    
    val sc = new SparkContext("local[*]" , "FilterLowListenCounts")
    
    val lines = sc.textFile("/Users/ekrem/No-cloud/datasets4senior/train_triplets_echonest.txt")
    
    val parsedLines = lines.map(parseLines)
    
    val filteredPlayCounts = parsedLines.filter(x => x._3 > 2)
    
    val userSongCount = filteredPlayCounts.map(x => (x._1)).countByValue().filter(x => x._2>20)
     
    //val results = filteredPlayCounts.collect()
    
    val pw = new PrintWriter(new File("NoLess2.txt"))
    
    var userCount = 0
    
    var playCount = 0
    
    for(result <- userSongCount) {
      
      userCount = userCount+1;
      
      playCount = playCount +result._2.toInt
      
      //pw.write(result._1 + " " + result._2 + " " + result._3 + "\t")
      pw.write(result._1 + " " + result._2 + "\n")
    }
    
    pw.close
    
    println("Finished " + playCount)
    
  }
}