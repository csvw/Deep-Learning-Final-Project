from data import VinBigDataset
from model import get_model
import torch
import gdcm
import time
from torch.utils.data import DataLoader
import math
from metrics import measure
from truth import create_ground
import matplotlib.pyplot as plt

def train():
  """
  Citation:
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    Pytorch Official Documentation, Torchvision Intermediate Tutorial
    Object Detection Finetuning Tutorial
  """
  if(torch.cuda.is_available()):
      device = torch.device("cuda")
      print(device, torch.cuda.get_device_name(0))
  else:
      device= torch.device("cpu")
      print(device)

  def collate(batch):
      return tuple(zip(*batch))

  s = time.time()
  
  model = get_model().to(device)
  dataset = VinBigDataset()
  
  TEST = True
  
  if TEST:
    keep_len = math.floor(.05 * len(dataset))
    throwaway = len(dataset) - keep_len

    keep, throwaway = torch.utils.data.random_split(dataset, [keep_len, throwaway])

    dlen = len(keep)
    train_len = math.floor(dlen * 0.8)
    val_len = dlen - train_len

    train, val = torch.utils.data.random_split(keep, [train_len, val_len])
    train_len_2 = math.floor(train_len * 0.8)
    test_len = train_len - train_len_2
    train, test = torch.utils.data.random_split(train, [train_len_2, test_len])
    train_loader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val, batch_size=2, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test, batch_size=2, shuffle=True, collate_fn=collate)
  else:
    dlen = len(dataset)
    train_len = math.floor(dlen * 0.8)
    val_len = dlen - train_len
    train, val = torch.utils.data.random_split(dataset, [train_len, val_len])
    train_len_2 = math.floor(train_len * 0.8)
    test_len = train_len - train_len_2
    train, test = torch.utils.data.random_split(train, [train_len_2, test_len])
    train_loader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val, batch_size=2, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test, batch_size=2, shuffle=True, collate_fn=collate)
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=0.0003, momentum=0.9, weight_decay=0.0005)
  lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
  
  create_ground(val_loader, dataset)
  
#   print(dataset.image_ids_inverse[dataset[0][1]["image_id"].item()])
  
  
#   test_val = torch.utils.data.Subset(dataset, [i for i in range(0, 16)])
#   test_train = torch.utils.data.Subset(dataset, [i for i in range(0, 16)])
  
#   test_train_loader = DataLoader(test_train, batch_size=2, shuffle=True, collate_fn=collate)
#   test_val_loader = DataLoader(test_val, batch_size=2, shuffle=True, collate_fn=collate)

  print("starting train")
  
  EPOCHS = 1
  itr = 1
  
  losses = []
  avg_loss = []

  for i in range(EPOCHS):
    model.train()
    for images, targets in train_loader:
      images = list(image.to(device) for image in images)
      targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
      loss = model(images, targets)
      loss = sum(l for l in loss.values())
      losses.append(loss)
      if itr > 11:
        avg_loss.append(sum(losses[itr-10:itr]) / 10)
      
      if itr % 1 == 0:
#         print(targets)
        print(f"Iteration #{itr} loss: {loss.item()}")
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      itr += 1
    lr_scheduler.step()
    
    
  print("Eval")
  model.eval()
  for images, targets in val_loader:
    images = list(image.to(device) for image in images)
    output = model(images)
    outputs = [{k:v.to(device) for k,v in t.items()} for t in output]
#     print(targets)
#     print(outputs)
    measure(targets, outputs, dataset)
    
    
  print("Time:")
  print(time.time() - s)
  
#   plt.plot(losses)
#   plt.savefig("Losses.png")
  
  plt.figure()
  plt.plot(avg_loss)
  plt.title("Average Training Loss")
  plt.xlabel("Iteration")
  plt.ylabel("Average Loss")
  plt.savefig("Loss 1 Epoch")
      
train()