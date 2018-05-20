#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 22:29:47 2018

@author: seha
"""

import numpy as np
import pandas as pd
from time import time
import taste_fn as fn
import matplotlib.pyplot as plt

triplets, users, songs = fn.load_files()

songs.plot.scatter(x = 'occurence', y = 'totalPlay')

users.plot.scatter(x = 'occurence', y = 'totalPlay')

plt.hist(np.log(songs['occurence']), bins = 20, color = 'red')
plt.xlabel('Occurence ln')
plt.ylabel('Frequency')
plt.show()