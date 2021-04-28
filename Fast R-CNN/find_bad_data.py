import pandas as pd
import os

train_csv = pd.read_csv(os.path.join('/home', 'jupyter', 'data', 'train.csv'))

x = train_csv[train_csv["x_max"] - train_csv["x_min"] < 20]
y = train_csv[train_csv["y_max"] - train_csv["y_min"] < 20]

print(x.head())
print(y.head())