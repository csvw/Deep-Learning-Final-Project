import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(pretrained=True, num_classes=16):
    """Loads a pretrained model and replaces
    the old heads (which were trained on Coco) with
    ones that have the appropriate number of classes.

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
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features_classifier = model.roi_heads.box_predictor.cls_score.in_features
    box_predictor = FastRCNNPredictor(in_features_classifier, num_classes).to(device)
    model.roi_heads.box_predictor = box_predictor

    return model