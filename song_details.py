#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 18:07:38 2018

@author: seha
"""

import tables
import pandas as pd

song_idx_dict = dict()
ss = dict()
def open_h5_file_read(h5filename):
    """
    Open an existing H5 in read mode.
    Same function as in hdf5_utils, here so we avoid one import
    """
    tables
    return tables.open_file(h5filename, mode='r')

def get_artist_name(h5,songidx=0):
    """
    Get artist name from a HDF5 song file, by default the first song in it
    """
    idx = song_idx_dict[songidx]
    return h5.root.metadata.songs.cols.artist_name[idx]

def get_release(h5,songidx=0):
    """
    Get release from a HDF5 song file, by default the first song in it
    """
    idx = song_idx_dict[songidx]
    return h5.root.metadata.songs.cols.release[idx]

def get_song_id(h5,songidx=0):
    """
    Get song id from a HDF5 song file, by default the first song in it
    """
    idx = song_idx_dict[songidx]
    return h5.root.metadata.songs.cols.song_id[idx]

def get_title(h5,songidx=0):
    """
    Get title from a HDF5 song file, by default the first song in it
    """
    idx = song_idx_dict[songidx]

    return h5.root.metadata.songs.cols.title[idx]

def get_track_id(h5,songidx=0):
    """
    Get track id from a HDF5 song file, by default the first song in it
    """
    print("hola")
    idx = song_idx_dict[songidx]
    print(idx)
    return h5.root.analysis.songs.cols.track_id[idx]

def fill_song_idx_dict(h5, song_dict):
    song_detail_dict = dict()
    for idx, val in enumerate(h5.root.metadata.songs.cols.song_id):
        str_val = str(val, 'utf-8')
        if str_val in song_dict:
            print(str_val , " " , idx)
            song_detail_dict[str_val] = idx
    return song_detail_dict
   
def start(song_dict):
    global song_idx_dict
    file_path = 'subset/msd_summary_file.h5'
    h5 = open_h5_file_read(file_path)
    song_idx_dict = fill_song_idx_dict(h5, song_dict)
    
    return h5, song_idx_dict

def stop(h5):
    h5.close()

def get_index_from_song_id(h5, song):
    for idx, val in enumerate(h5.root.metadata.songs.cols.song_id):
        if val == song:
            return idx
        
    return 0

