# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 20:50:35 2020

@author: cling
"""

import random
import numpy as np
import pandas as pd


# Pre-processing IMDB dataset
mat = pd.read_csv('IMDB/movie_metadata.csv')

d_names, m_names, a_names = list(mat.director_name), list(mat.movie_title), list(mat.actor_1_name)
names = set(d_names+m_names+a_names)
names.remove(np.nan)
names = list(names)
indices = list(range(len(names)))
name_dir = dict(zip(names, indices))

for index, row in mat.iterrows():
    try:
        d_name = str(name_dir[row.director_name])
        a_name = str(name_dir[row.actor_1_name])
        m_name = str(name_dir[row.movie_title])
    
        with open('IMDB/link.dat', 'a') as w:
            w.writelines(m_name+' '+d_name+'\n')
            w.writelines(m_name+' '+a_name+'\n')
            w.writelines(a_name+' '+d_name+'\n')
    except:
        continue

with open('IMDB/node.dat', 'a') as w:
    for i, x in enumerate(names):
        if x in d_names:
            w.writelines(str(i)+' 0\n')
        elif x in m_names:
            w.writelines(str(i)+' 1\n')
        elif x in a_names:
            w.writelines(str(i)+' 2\n')
    