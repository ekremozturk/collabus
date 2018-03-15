package com.collabus.msd.spark

import org.hdfgroup.spark.hdf5._
import org.apache.spark.sql.SparkSession
import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import java.io._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._


object HDF5read {
  
  def main(args: Array[String]) {
    
    Logger.getLogger("org").setLevel(Level.ERROR)
    
    val spark = SparkSession
      .builder()
      .appName("Spark SQL HDF5 MSD")
      .master("local[*]")
      .getOrCreate()
      
    val t_start = System.nanoTime()
    
    val h5_files_path = "sparky://files" //Input for virtual path of files
    
    val file_path = "/Volumes/Expansion-Drive/MSD/A/A/A" //Actual dataset file path
    
    //Reads the files recursively to extract file paths
    val df_files = spark.read.option("extension", "h5")
      .option("recursion", "true")
      .hdf5(file_path, h5_files_path) 
    
    //Spark action
    val files = df_files.select("FilePath").collect()
    
    //Traverse collected tracks
    for(track <- files){
      
      val track_path = track.getString(0)
      
      val current_track = HDF5getters.to_triplet(track_path)       
      
      println("")
      println("Track with ID "+ current_track.track_id)
      println(current_track.artist_terms.show)
      println(current_track.similar_artists.show)
      
    }
   
    val t_end = System.nanoTime()
    
    println((t_end-t_start)/1000000000.0 + " seconds")
    
  }

}