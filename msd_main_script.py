#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 17:13:31 2018

@author: ekrem
"""

import hdf5_getters as hdf
import time
h5 = hdf.open_h5_file_read('/Volumes/Expansion-Drive/MSD/AdditionalFiles/msd_summary_file.h5')

songs = []

start_time = time.time()

for i in range (10000):
  track_id = hdf.get_track_id(h5, i)
  artist_id = hdf.get_artist_id(h5, i)
  artist_name = hdf.get_artist_name(h5, i)
  song_id = hdf.get_song_id(h5, i)
  title = hdf.get_title(h5, i)
  year = hdf.get_year(h5, i)
  
  song = [track_id, song_id, artist_id, title, artist_name, year]
  
  songs.append(song)

end_time = time.time()

print(end_time-start_time)
h5.close()
