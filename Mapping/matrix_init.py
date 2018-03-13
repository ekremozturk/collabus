
import numpy as np
import scipy.sparse as sps
import pandas as pd


song_dict = dict()
user_dict = dict()


with open("echonest_only_songids.txt") as file:
    content = file.readlines()

content =[x.strip() for x in content]

count = 0
for item in content:
    song_dict[item] = count
    count += 1 

with open("echonest_only_userids.txt") as file:
    content = file.readlines()

content =[x.strip() for x in content]

count = 0
for item in content:
    user_dict[item] = count
    count += 1 
    
record = sps.lil_matrix((len(user_dict), len(song_dict) ), dtype=np.int32)


with open("train_triplets.txt") as infile:
    for line in infile:
        items = line.split()
        user_no = user_dict[items[0]]
        song_no = song_dict[items[1]]
        listening_count = items[2]
        record[user_no,song_no] = listening_count 
