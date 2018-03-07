#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:59:06 2018

@author: ekrem
"""

import numpy as np
import matplotlib.pyplot as plt

file_path = "/Users/ekrem/No-cloud/datasets4senior/echonest_scaled_playcounts.txt"

file = open(file_path, "r")

play_counts = []

for line in file:
  play_counts.append(int(line))

#total number of entries in dataset
play_count_entries_total = len(play_counts)

#the ranges of bins
bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 25.5, 50.5, 9967.5]

#it takes time to process
histogram = plt.hist(play_counts, bins=bins)
plt.title("Histogram of Play Counts")
plt.show()

play_counts_in_range = histogram[0]
ranges = histogram[1]

play_counts_percentage = []

for count in play_counts_in_range:
  play_counts_percentage.append(count/play_count_entries_total)

file.close()