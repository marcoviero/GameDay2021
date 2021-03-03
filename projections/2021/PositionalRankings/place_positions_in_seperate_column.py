#!/usr/bin/env python

import pdb
import sys
import os
import pandas as pd

filein = sys.argv
print(filein)
orig_df = pd.read_excel(filein[1],index_col='RANK')
new_df = orig_df.copy()
new_df.rename(columns = {"VS. ADP": "Elig. Pos."},inplace=True)
new_df.rename(index={'RANK':'Rank'}, inplace=True)
for i,data in orig_df.iterrows():
    player = data[0].replace('\xa0','').split('(')[0]
    positions = data[0].replace('\xa0','').split('(')[1].split(')')[0]
    if '-' in positions:
        positions = positions.split('- ')[1].replace(',','/')
    else:
        positions = positions.replace(',','/')
    
    #new_df.loc[i,:] = [player, positions]
    new_row = [player]+(data[1:-1].to_list())+[positions]
    new_df.loc[i,:] = new_row

pdb.set_trace()
new_df.to_excel(filein[1].split('.')[0]+'.xlsx')
