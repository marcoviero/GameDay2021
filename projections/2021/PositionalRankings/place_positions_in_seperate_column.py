#!/usr/bin/env python

import pdb
import sys
import os
import pandas as pd

filein = sys.argv
print(filein)
orig_df = pd.read_excel(filein[1],index_col='Rank')
new_df = orig_df.copy()
for i,data in orig_df.iterrows():
    player = data[0].split('(')[0]
    positions = data[0].split('(')[1].split(')')[0]
    if '-' in positions:
        positions = positions.split('- ')[1].replace(',','/')
    else:
        positions = positions.replace(',','/')
    
    new_df.loc[i,:] = [player, positions]

pdb.set_trace()
new_df.to_excel(filein[1].split('.')[0]+'.xlsx')
