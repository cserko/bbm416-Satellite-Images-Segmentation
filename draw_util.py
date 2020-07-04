import time
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv
from read_data import system_load
colab = False
try:
  from google.colab.patches import cv2_imshow
  colab = True
except Exception as e:
  print(e)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor.squeeze().numpy()

    
    
def draw_boxes(images, targets):
    color = (0, 255, 0)
    thickness = 2
    un = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    nn_image = un(images)
    nn_image = np.transpose(nn_image, (1, 2, 0))
    #print(targets[0]["id"])
    #drew = cv.imread(targets[0]["id"][0])
    drew = nn_image.copy()

    #print(drew)
    #print(targets[0]["labels"])
    label_c = 0
    for i in targets[0]["boxes"]:
        i = [int(j) for j in i]
        #print(i)
        pt1 =(i[0], i[1])
        pt2 =(i[2], i[3])
        cv.rectangle(drew, pt1, pt2, color, thickness)
        #targets[0]["labels"][label_c]
        drew = cv.putText(drew, str(targets[0]["labels"].tolist()[label_c]), pt1,  cv.FONT_HERSHEY_PLAIN, 1, \
                    (255,140,255),thickness=1) # Write the prediction class
        label_c += 1
    
    imshow(drew)
    

    
def draw(img, tup, color, thickness):
    cv.rectangle(img, (i[0], i[1]), (i[2], i[3]), color, thickness)
    
    return img

def imshow(img):
    #img = np.transpose(img, (1, 2, 0))
    #print(img.shape)
    if colab is True:
      cv2_imshow(ResizeWithAspectRatio(img))
    else:
      cv.imshow("imgi", ResizeWithAspectRatio(img))
    
    cv.waitKey(0)
    
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)