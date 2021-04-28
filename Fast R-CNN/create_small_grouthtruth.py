import pandas as pd
import os
from data import VinBigDataset


dataset = VinBigDataset(fname="small_train.csv")

if not os.path.exists('/home/jupyter/data/groundtruths'):
    os.makedirs('/home/jupyter/data/groundtruths')

for index, row in dataset.train_labels.iterrows():
  with open("/home/jupyter/data/groundtruths/" + str(row.name) + ".txt", "a") as f:
    if row["x_max"] != row["x_max"]:
      f.write("{} {} {} {} {}\n".format(
        row['class_name'],
        0,
        0,
        255,
        255
      ))
    else: 
      f.write("{} {} {} {} {}\n".format(
        row['class_name'],
        row['x_min'],
        row['y_min'],
        row['x_max'] - row['x_min'],
        row['y_max'] - row['y_min']
      ))