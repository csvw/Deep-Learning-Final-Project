import os

def create_ground(subset, data, im_size=255):
  if not os.path.exists('/home/jupyter/data/groundtruths'):
      os.makedirs('/home/jupyter/data/groundtruths')

  for row, target in subset:
#     print(target)
#     print(len(target))
    target = target[0]
    im_id = data.image_ids_inverse[target['image_id'].item()]
    for i in range(len(target["boxes"])):
      cname = data.class_id_inverse[target["labels"][i].item()]
      xmin = target["boxes"][i][0]
      ymin = target["boxes"][i][1]
      xmax = target["boxes"][i][2]
      ymax = target["boxes"][i][3]
#       print(xmin, ymin, xmax, ymax)
#       print(xmin)
      with open("/home/jupyter/data/groundtruths/" + str(im_id) + ".txt", "a") as f:
        if xmin != xmin:
          f.write("{} {} {} {} {}\n".format(
            cname,
            0,
            0,
            im_size,
            im_size
          ))
        else: 
          f.write("{} {} {} {} {}\n".format(
            cname,
            xmin,
            ymin,
            xmax - xmin,
            ymax - ymin
          ))