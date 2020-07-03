import time
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from classes import class_list



class X_View_FasterRCNN(Dataset):
    def __init__(self, meta, root_dir, include=None, classes=None, transforms=None):
        self.meta = meta
        self.root = root_dir
        self.transforms = transforms
        self.meta_keys = list(meta.keys())
        self.classes = classes
        self.include = [i for i in include if i in self.classes.keys()]
        self.ordered_class = {include[i]:i for i in range(len(include))}
        self.not_found_images = []
        self.os_trucated_error = []
        self.generic_error = []
        print("correspondence of class names and labels:", self.ordered_class)
        
    def __getitem__(self, idx):
        # load images ad masks
        img_name = self.meta_keys[idx]
        img_path = os.path.join(self.root, "train_images", img_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            self.not_found_images.append(img_path)
            return None
        except OSError:
            self.os_trucated_error.append(img_path)
            return None
        except Exception as e:
            self.generic_error.append(e)
            return None



        boxes = []
        labels = []
        img_meta = self.meta[img_name]
        for i in range(len(img_meta)):
            
            if int(img_meta[i][1]) in self.include:
                if img_meta[i][0][0] < 0 or img_meta[i][0][1] < 0 or img_meta[i][0][2] < 0 or img_meta[i][0][3] < 0:
                    continue
                #print("box:", img_meta[i][0], "label", img_meta[i][1])
                ordered_label = self.ordered_class[img_meta[i][1]]
                boxes.append(img_meta[i][0])
                labels.append(ordered_label)
        if len(labels) < 2:
            return None
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        #labels = torch.ones((num_objs,), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        #area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.meta)
    
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
      batch = [[], []]
    return torch.utils.data.dataloader.default_collate(batch)
    
def load_dataset(meta, root_dir, include=None, workers=0):
    '''Train'''
    
    classes = class_list()
    
    #transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = X_View_FasterRCNN(meta=meta, root_dir=root_dir, include=include, classes=classes, transforms=transform)
    train_size = int(len(dataset) * 0.8)
    validation_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - validation_size
    # this is a risky operation
    train, validate, test = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])
    #changing batch_size & workers
    loader_train = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=workers)
    loader_validate = torch.utils.data.DataLoader(validate, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=workers)
    loader_test = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=workers)

    datasets = {"train": loader_train, "validate": loader_validate, "test": loader_test}
    dataset_lengths = {"train": len(loader_train), "validate": len(loader_validate), "test":len(loader_test)}
    
    return datasets, dataset_lengths