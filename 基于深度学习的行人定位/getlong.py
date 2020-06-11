import pandas as pd
import numpy as np
df_train = pd.read_csv('train.csv')
X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
width = X_train[:,2]-X_train[:,0]
long = X_train[:,3]-X_train[:,1]
#x = np.column_stack((1/width,1/long))
data1 = pd.DataFrame(width) # header:原第一行的索引，index:原第一列的索引
data1.to_csv('result1.csv')

