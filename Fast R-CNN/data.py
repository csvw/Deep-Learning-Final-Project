import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms as T

class VinBigDataset(object):
    """Constructs a Pytorch Dataset object from the
    VinBigData dataset, extracts the pixel array
    objects from the dicom files and extracts labels
    from train.csv.

    Citation:
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    Pytorch Official Documentation, Torchvision Intermediate Tutorial
    Object Detection Finetuning Tutorial
    """
    def __init__(self, transforms=None, fname=None, im_size=255):
      self.im_size = im_size
      self.transforms = transforms
      if transforms is None:
        self.transforms = [T.ToTensor()]
      self.root = os.path.join('/home/jupyter', 'data')
      self.train_path = os.path.join(self.root, 'train')
      self.train = list(sorted(os.listdir(self.train_path)))
      if fname is None:
        self.train_labels = pd.read_csv(os.path.join(self.root, 'fixed_train.csv'))
      else:
        self.train_labels = pd.read_csv(os.path.join(self.root, fname))
      self.train_labels = self.train_labels.set_index('image_id')
      temp = self.train_labels.copy(deep=True)
      temp = temp[~temp.index.duplicated(keep='first')]
      self.image_ids = {temp.index[i]: i for i in range(len(temp))}
      self.image_ids_inverse = {i: temp.index[i] for i in range(len(temp))}
      self.class_id_inverse = {}
      
      for cname in self.train_labels["class_name"].unique():
        class_id = self.train_labels.loc[self.train_labels["class_name"] == cname, "class_id"].values[0]
        self.class_id_inverse[class_id] = cname


    def __getitem__(self, idx):
      path = os.path.join(self.train_path, self.train[idx])
      img = Image.open(path)
      for t in self.transforms:
        img = t(img)
      img_id = self.train[idx].split('.')[0]
      target = {}
      boxes = []
      labels = []
      area = []
      for i, row in self.train_labels.loc[img_id].iterrows():
#         print(row)
        xmin = row['x_min'] * self.im_size
        ymin = row['y_min'] * self.im_size
        xmax = row['x_max'] * self.im_size
        ymax = row['y_max'] * self.im_size
        if xmax != xmax:
          xmin = 0
          ymin = 0
          xmax = self.im_size
          ymax = self.im_size
        box = []
        box.append(xmin)
        box.append(ymin)
        box.append(xmax)
        box.append(ymax)
        labels.append(row['class_id'])
        area.append((box[2] - box[0]) * (box[3] - box[1]))
        if not box in boxes:
          boxes.append(box)
                
      #These are required by the Fast RCNN API
      target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
      target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
      target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
      target["image_id"] = torch.tensor([self.image_ids[img_id]], dtype=torch.int64)
      target["area"] = torch.as_tensor(area, dtype=torch.float32)

      return img, target

    def __len__(self):
      return len(self.train)