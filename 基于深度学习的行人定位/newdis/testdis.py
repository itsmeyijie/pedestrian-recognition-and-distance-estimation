import torch
import pandas as pd
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
import os
import cv2
from sklearn.preprocessing import StandardScaler
PATH = "./test.pkl"
model = torch.load(PATH)
model.eval()

df = pd.DataFrame(columns=['filename', 'class',  \
                           'z', 'z_p',   \
                           'xmin', 'ymin', 'xmax', 'ymax'])

df_r = pd.read_csv("pttest.csv")
df_test = pd.read_csv('ptresult.csv')
filename = df_r[['filename','zloc']].values
X_test = df_r[['xmin', 'ymin', 'xmax', 'ymax']].values
#df_result = pd.read_csv("pttestresult.csv")
y_test = df_r[['zloc']].values
#print(X_test)
scalar2 = StandardScaler()
X_test = scalar2.fit_transform(X_test)
Y_test = scalar2.fit_transform(y_test)
x_t = torch.tensor(X_test,dtype=torch.float32)
x_tt = Variable(x_t)
y_pred_tt = model.forward(x_tt)
y_pred_test = scalar2.inverse_transform(y_pred_tt.detach().numpy())
print(y_pred_test)
zz = np.hstack((filename,df_r[['xmin', 'ymin', 'xmax', 'ymax']].values))
xx =np.hstack((zz, y_pred_test))
save = pd.DataFrame(xx , columns = ['filename','zloc','xmin', 'ymin', 'xmax', 'ymax','zloc_pred'])

save.to_csv('pt.csv',index=False)



