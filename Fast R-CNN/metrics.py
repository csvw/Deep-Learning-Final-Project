import torch
import os

def measure(targets, outputs, data):
  if not os.path.exists('/home/jupyter/data/detections'):
    os.makedirs('/home/jupyter/data/detections')
  
  for target, output in zip(targets, outputs):
    im_id = data.image_ids_inverse[target["image_id"].item()]
#     print(output)
    for i in range(len(output["boxes"])):
#       print(i)
      with open("/home/jupyter/data/detections/" + im_id + ".txt", "a") as f:
        f.write("{} {} {} {} {} {}\n".format(
          data.class_id_inverse[output['labels'][i].item()],
          output['scores'][i],
          output['boxes'][i][0],
          output['boxes'][i][1],
          output['boxes'][i][2] - output['boxes'][i][0],
          output['boxes'][i][3] - output['boxes'][i][1]
        ))

      
     
  
