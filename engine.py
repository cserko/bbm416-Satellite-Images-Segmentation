import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

#from coco_utils import get_coco_api_from_dataset
#from coco_eval import CocoEvaluator
import utils
from draw_util import draw_boxes


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for i in metric_logger.log_every(data_loader, print_freq, header):

        try:
            images, targets = i
            '''Burası değiştirilecek'''
            targets["boxes"] = targets["boxes"].to(device)
            targets["labels"] = targets["labels"].to(device)
            targets["boxes"].squeeze_()
            targets["labels"].squeeze_()
            targets1 = [{k: v for k, v in targets.items()}]
            
            images = images.to(device)
            targets = targets1
            # zero the parameter gradients

            # forward
            # track history if only in train
            #images = list(image.to(device) for image in images)
            #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()
            #print(targets[0]["boxes"])
            if not math.isfinite(loss_value):
                print(images.size())
                print(targets[0]["boxes"])
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        except ValueError:
            continue
            
    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, draw=False):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    #coco = get_coco_api_from_dataset(data_loader.dataset)
    #iou_types = _get_iou_types(model)
    #coco_evaluator = CocoEvaluator(coco, iou_types)

    for i in metric_logger.log_every(data_loader, 100, header):

        try:
            images, targets = i
            targets["boxes"] = targets["boxes"].to(device)
            targets["labels"] = targets["labels"].to(device)
            targets["boxes"].squeeze_()
            targets["labels"].squeeze_()
            
            targets1 = [{k: v.to(device) for k, v in targets.items()}]

            images = list(img.to(device) for img in images)
            targets = targets1

        except ValueError:
            continue    
        

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        #outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        print(outputs)
        outputs = [{k: v.to(cpu_device) for k, v in outputs[0].items()}]
        if draw:
            draw_boxes(images[0].to(cpu_device), targets)
        model_time = time.time() - model_time

        #res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    #coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    #coco_evaluator.accumulate()
    #coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    #return coco_evaluator
    return None