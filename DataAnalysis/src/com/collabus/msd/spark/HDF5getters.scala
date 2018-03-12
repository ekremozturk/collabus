package com.collabus.msd.spark

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import java.io._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.hdfgroup.spark.hdf5._

object HDF5getters {
  
  case class Track(track_id: String, artist_terms: DataFrame, similar_artists: DataFrame)
  
  def to_triplet(file_path: String): Track = {
    
    val file_path_fields = file_path.split("/")
    
    val track_id = file_path_fields(7).substring(0, file_path_fields(7).length()-3)
    
    val artist_terms = join_artist_terms(
        get_artist_terms(file_path),
        get_artist_terms_freq(file_path),
        get_artist_terms_weight(file_path))
        
    val similar_artists = get_similar_artists(file_path).drop("FileID", "Index").toDF("Similar Artists")
    
    val track: Track = Track(track_id, artist_terms, similar_artists)
    
    return track
  }
  
  val spark = SparkSession
      .builder()
      .appName("Spark SQL HDF5 MSD")
      .master("local[*]")
      .getOrCreate()
  
  def join_artist_terms(artist_terms: DataFrame, 
      artist_terms_freq: DataFrame, 
      artist_terms_weight: DataFrame): DataFrame = {
    
    val iv_artist_terms = artist_terms.select("Index", "Value")
    val iv_artist_terms_freq = artist_terms_freq.select("Index", "Value")
    val iv_artist_terms_weight = artist_terms_weight.select("Index", "Value")
    
    val joined_artist_terms = iv_artist_terms.join(iv_artist_terms_freq, "Index")
                                             .join(iv_artist_terms_weight, "Index")
                                             .orderBy("Index")
                                             .drop("Index")
                                             .toDF("Term", "Frequency", "Weight")
    return joined_artist_terms
  }
  
  def get_artist_terms(file_path: String): DataFrame = {
    
    val df = spark.read.option("extension", "h5")
                  .hdf5(file_path, "/metadata/artist_terms")                        
    
    return df                        
  }
  
  def get_artist_terms_freq(file_path: String): DataFrame = {
    
    val df = spark.read.option("extension", "h5")
                  .hdf5(file_path, "/metadata/artist_terms_freq")                        
    
    return df                        
  }
  
  def get_artist_terms_weight(file_path: String): DataFrame = {
    
    val df = spark.read.option("extension", "h5")
                  .hdf5(file_path, "/metadata/artist_terms_weight")                        
    
    return df                        
  }
  
  def get_similar_artists(file_path: String): DataFrame = {
    
    val df = spark.read.option("extension", "h5")
                  .hdf5(file_path, "/metadata/similar_artists")
    
    return df                        
  }
  
  def get_artist_mbtags(file_path: String): DataFrame = {
    
    if(get_artist_mbtags_count(file_path).count() == 0){
      return get_artist_mbtags_count(file_path)
    }
    
    val df = spark.read.option("extension", "h5")
                  .hdf5(file_path, "/musicbrainz/artist_mbtags")                        
    
    return df                        
  }
  
  def get_artist_mbtags_count(file_path: String): DataFrame = {
    
    val df = spark.read.option("extension", "h5")
                  .hdf5(file_path, "/musicbrainz/artist_mbtags_count")                        
    
    return df                        
  }
  
}