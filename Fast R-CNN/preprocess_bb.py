import pandas as pd
import os

train_csv = pd.read_csv(os.path.join('/home', 'jupyter', 'data', 'train.csv'))
meta_csv = pd.read_csv(os.path.join('/home', 'jupyter', 'data', 'train_meta.csv'))

train_csv = train_csv.sort_index()
meta_csv = meta_csv.sort_index()

train_csv = train_csv.sort_values(by="image_id")
meta_csv = meta_csv.sort_values(by="image_id")

temp = train_csv[(train_csv["x_max"] - train_csv["x_min"] < 20)]
temp = train_csv[(train_csv["y_max"] - train_csv["y_min"] < 20)]

for i, row in temp.iterrows():
  if os.path.exists("/home/jupyter/data/train/" + str(row["image_id"]) + ".png"):
    os.rename("/home/jupyter/data/train/" + str(row["image_id"]) + ".png", "/home/jupyter/data/removed/" + row["image_id"] + ".png")
    
ids = [
  "65ad4fb69f36c807fce87e66a1c6533d",
  "74464c6b0f2b89fa3e8d7262571d86c8",
  "9f6515eac1d5043d511068bd757a17e1",
  "a9519035bae20b2267b38445724221a2",
  "c29cec7b8f63ab34b1c88153b3efd4df"
]

for i in ids:
  if os.path.exists("/home/jupyter/data/train/" + str(i) + ".png"):
    os.rename("/home/jupyter/data/train/" + str(i) + ".png", "/home/jupyter/data/removed/" + str(i) + ".png")

meta_csv = meta_csv[~(train_csv["x_max"] - train_csv["x_min"] < 20)]
train_csv = train_csv[~(train_csv["x_max"] - train_csv["x_min"] < 20)]
meta_csv = meta_csv[~(train_csv["y_max"] - train_csv["y_min"] < 20)]
train_csv = train_csv[~(train_csv["y_max"] - train_csv["y_min"] < 20)]

train_csv["class_id"] += 1
train_csv["class_name"] = train_csv["class_name"].replace(["No finding"], "No_finding")
train_csv["class_name"] = train_csv["class_name"].replace(["Aortic enlargement"], "Aortic_enlargement")
train_csv["class_name"] = train_csv["class_name"].replace(["Pleural thickening"], "Pleural_thickening")
train_csv["class_name"] = train_csv["class_name"].replace(["Pulmonary fibrosis"], "Pulmonary_fibrosis")
train_csv["class_name"] = train_csv["class_name"].replace(["Lung Opacity"], "Lung_Opacity")
train_csv["class_name"] = train_csv["class_name"].replace(["Other lesion"], "Other_lesion")
train_csv["class_name"] = train_csv["class_name"].replace(["Pleural effusion"], "Pleural_effusion")
train_csv["class_name"] = train_csv["class_name"].replace(["Nodule/Mass"], "Nodule_mass")

train_csv = train_csv[train_csv['image_id'] != "65ad4fb69f36c807fce87e66a1c6533d"]
train_csv = train_csv[train_csv['image_id'] != "74464c6b0f2b89fa3e8d7262571d86c8"]
train_csv = train_csv[train_csv['image_id'] != "9f6515eac1d5043d511068bd757a17e1"]
train_csv = train_csv[train_csv['image_id'] != "a9519035bae20b2267b38445724221a2"]
train_csv = train_csv[train_csv['image_id'] != "c29cec7b8f63ab34b1c88153b3efd4df"]

meta_csv = meta_csv.set_index('image_id')

for index, row in train_csv.iterrows():
  im_id = row['image_id']
  if im_id in meta_csv.index:
    train_csv.loc[index, 'x_min'] = train_csv.loc[index, 'x_min'] / meta_csv.at[im_id, 'dim1']
    train_csv.loc[index, 'y_min'] = train_csv.loc[index, 'y_min'] / meta_csv.at[im_id, 'dim0']
    train_csv.loc[index, 'x_max'] = train_csv.loc[index, 'x_max'] / meta_csv.at[im_id, 'dim1']
    train_csv.loc[index, 'y_max'] = train_csv.loc[index, 'y_max'] / meta_csv.at[im_id, 'dim0']
  else:
    print(im_id)




train_csv.to_csv("/home/jupyter/data/fixed_train.csv")
meta_csv.to_csv("/home/jupyter/data/fixed_meta.csv")
