from data import VinBigDataset
from model import get_model
import torch
import gdcm
import time

if(torch.cuda.is_available()):
    device = torch.device("cuda")
    print(device, torch.cuda.get_device_name(0))
else:
    device= torch.device("cpu")
    print(device)

def collate(batch):
    return tuple(zip(*batch))

s = time.time()
  
model = get_model()
model.to(device)
dataset = VinBigDataset()
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)
images, targets = next(iter(loader))
images = list(image.to(device) for image in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
print("Before Run")
print(time.time() - s)
output = model(images,targets)   # Returns losses and detections
print("After Run")
print(time.time() - s)
print(output)
model.eval()
# x = [torch.rand(1, 300, 400).to(device), torch.rand(1, 500, 400).to(device)]
predictions = model(images)
print(predictions)
print("After Eval")
print(time.time() - s)