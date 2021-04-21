import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model(pretrained=True, num_classes=14):
    """Loads a pretrained model and replaces
    the old heads (which were trained on Coco) with
    ones that have the appropriate number of classes.

    Citation:
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    Pytorch Official Documentation, Torchvision Intermediate Tutorial
    Object Detection Finetuning Tutorial
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

    in_features_classifier = model.roi_heads.box_predictor.cls_score.in_features
    box_predictor = FastRCNNPredictor(in_features_classifier, num_classes)
    model.roi_heads.box_predictor = box_predictor

    return model
