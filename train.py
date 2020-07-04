
import psutil
import humanize
import os
import GPUtil as GPU

from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN

import numpy as np
import torch, torchvision
from engine import train_one_epoch, evaluate
import utils

def avg_acc(sozluk):
    return np.average(np.array([i for i in sozluk.values()]))
    
def printm():
    GPUs = GPU.getGPUs()
    # XXX: only one GPU on Colab and isn’t guaranteed
    gpu = GPUs[0]

    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available), " |     Proc size: " + humanize.naturalsize(process.memory_info().rss))
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total     {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))


def train_model(model, dataloaders, dataset_lengths, optimizer, scheduler, num_epochs, device="cpu"):
    print("start training. . .")

    for epoch in range(num_epochs):
        try:
          
            train_one_epoch(model, optimizer, dataloaders["train"], device, epoch, print_freq=50)
            scheduler.step()

        except Exception as e:
          print(e)
          
    return model

def model_creation(pretrain=True, num_classes=5, num_epoch=20, device="cuda:0"):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 5
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)
    model = FasterRCNN(backbone,
                    num_classes=num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)
    if device == 'cuda:0':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            print("cuda is no available")
    
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 50

    return model, optimizer, lr_scheduler, num_epochs, 

def load_model(model_path):
    return torch.load(model_path)