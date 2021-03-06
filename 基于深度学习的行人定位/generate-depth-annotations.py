'''
Purpose: Generate dataset for depth estimation
'''
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

df = pd.read_csv('annotations.csv')
new_df = df.loc[df['class'] != 'DontCare']
result_df = pd.DataFrame(columns=['filename', 'xmin', 'ymin', 'xmax', 'ymax', \
                           'angle', 'xloc', 'yloc', 'zloc'])

pbar = tqdm(total=new_df.shape[0], position=1)

for idx, row in new_df.iterrows():
    pbar.update(1)
     #labels  ->original_data/train_annots
    if os.path.exists(os.path.join("original_data/train_annots", row['filename'])):
        result_df.at[idx, 'filename'] = row['filename']
        #print(row)
        result_df.at[idx, 'xmin'] = row['xmin']
        result_df.at[idx, 'ymin'] = row['ymin']
        result_df.at[idx, 'xmax'] = row['xmax']
        result_df.at[idx, 'ymax'] = row['ymax']

        result_df.at[idx, 'angle'] = row['observation angle']
        result_df.at[idx, 'xloc'] = row['xloc']
        result_df.at[idx, 'yloc'] = row['yloc']
        result_df.at[idx, 'zloc'] = row['zloc']

mask = np.random.rand(len(result_df)) < 0.9
train = result_df[mask]
test = result_df[~mask]
#print(train)
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
