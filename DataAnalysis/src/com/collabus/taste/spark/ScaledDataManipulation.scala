package com.collabus.taste.spark

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import java.io._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

object ScaledDataManipulation {
  
  def main(args: Array[String]) {
  
  Logger.getLogger("org").setLevel(Level.ERROR)
    
    // Use new SparkSession interface in Spark 2.0
    val spark = SparkSession
      .builder
      .appName("SparkSQL")
      .master("local[*]")
      .getOrCreate()
    
    //Read the file
    val lines = spark.sparkContext.textFile("/Users/ekrem/No-cloud/datasets4senior/echonest_scaled_playcounts.txt").map(row=>row.toInt)
    
    import spark.implicits._
    val schemaLines = lines.toDS
    schemaLines.createOrReplaceTempView("lines")
    
    schemaLines.printSchema()
    
    val orderedPlays = spark.sql("SELECT * FROM lines ORDER BY value DESC")
    
    //val topPlays = orderedPlays.filter(orderedPlays("value")>400)
    
    //val topPlaysResult = topPlays.collect
    
    //topPlaysResult.foreach(row => println(row.get(0)))
    
    spark.stop()
  }
  
}