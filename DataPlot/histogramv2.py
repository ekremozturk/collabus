#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:47:39 2018

@author: seha
"""

import numpy as np
import scipy.sparse as sps
import pandas as pd



with open("/home/seha/workspace/listening_matrix_extractor/echonest_only_userids.txt" )as file:
    content = file.readlines()

content =[x.strip() for x in content]

songs = np.empty(len(content), dtype ='object')

count = 0
for item in content:
    songs[count] = item
    count += 1 
    
    
listen_counts = np.zeros((songs.size,), dtype=int)

with open("/home/seha/workspace/listening_matrix_extractor/train_triplets.txt") as infile:
  for line in infile:
      (_,song_id,listen_count) = line.split()
      index = np.where(songs == song_id)
      listen_counts[index[0]] = listen_counts[index[0]] + int (listen_count)
      
