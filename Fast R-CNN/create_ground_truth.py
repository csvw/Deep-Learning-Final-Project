import pandas as pd
import os

train_csv = pd.read_csv(os.path.join('/home', 'jupyter', 'data', 'train.csv'))

train_csv["class_id"] += 1
train_csv["class_name"] = train_csv["class_name"].replace(["No finding"], "No_finding")
train_csv["class_name"] = train_csv["class_name"].replace(["Aortic enlargement"], "Aortic_enlargement")
train_csv["class_name"] = train_csv["class_name"].replace(["Pleural thickening"], "Pleural_thickening")
train_csv["class_name"] = train_csv["class_name"].replace(["Pulmonary fibrosis"], "Pulmonary_fibrosis")
train_csv["class_name"] = train_csv["class_name"].replace(["Lung Opacity"], "Lung_Opacity")
train_csv["class_name"] = train_csv["class_name"].replace(["Other lesion"], "Other_lesion")
train_csv["class_name"] = train_csv["class_name"].replace(["Pleural effusion"], "Pleural_effusion")
train_csv["class_name"] = train_csv["class_name"].replace(["Nodule/Mass"], "Nodule_mass")



print(train_csv["image_id"].unique())

print(train_csv.pivot_table(index=['image_id'], aggfunc='size'))

train_csv = train_csv.set_index('image_id')
train_csv = train_csv.sort_index()
train_csv.to_csv("/home/jupyter/data/fixed_train.csv")

small_train = train_csv.loc[train_csv.index <= "004d3"]
small_train.to_csv("/home/jupyter/data/small_train.csv")

if not os.path.exists('/home/jupyter/data/groundtruths'):
    os.makedirs('/home/jupyter/data/groundtruths')

for index, row in train_csv.iterrows():
  with open("/home/jupyter/data/groundtruths/" + str(row['image_id']) + ".txt", "a") as f:
    if row["x_max"] != row["x_max"]:
      f.write("{} {} {} {} {}\n".format(
        row['class_name'],
        0,
        0,
        1.0,
        1.0
      ))
    else: 
      f.write("{} {} {} {} {}\n".format(
        row['class_name'],
        row['x_min'],
        row['y_min'],
        row['x_max'] - row['x_min'],
        row['y_max'] - row['y_min']
      ))