import os
import numpy as np
import torch
import dicom

class VinBigDataset(object):
    """Constructs a Pytorch Dataset object from the
    VinBigData dataset and configures it for compatibility
    with the Mask R-CNN package.

    Citation:
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    Pytorch Official Documentation, Torchvision Intermediate Tutorial
    Object Detection Finetuning Tutorial
    """
    def __init__(self, root, transforms, train, test):
        self.root = root
        self.transforms = transforms
        self.train = list(sorted(os.listdir(os.path.join(root, train)))) # TODO: Set the path correctly
        self.test  = list(sorted(os.listdir(os.path.join(root, test)))) # TODO: Set path correctly

    def __getitem__(self, idx):
