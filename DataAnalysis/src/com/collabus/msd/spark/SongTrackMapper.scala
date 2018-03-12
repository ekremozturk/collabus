package com.collabus.msd.spark

import org.hdfgroup.spark.hdf5._
import org.apache.spark.sql.SparkSession
import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import java.io._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

object SongTrackMapper {
  
  def main(args: Array[String]) {
    
    Logger.getLogger("org").setLevel(Level.ERROR)
    
    val spark = SparkSession
      .builder()
      .appName("Spark SQL HDF5 MSD")
      .master("local[*]")
      .getOrCreate()
      
    
  }
  
  
  
}