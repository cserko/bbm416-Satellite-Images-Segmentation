import time
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class X_View_FasterRCNN(Dataset):
    def __init__(self, meta, root_dir, transforms=None):
        self.meta = meta
        self.root = root_dir
        self.transforms = transforms
        self.meta_keys = list(meta.keys())

    def __getitem__(self, idx):
        # load images ad masks
        img_name = self.meta_keys[idx]
        img_path = os.path.join(self.root, "train_images", img_name)
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        img_meta = self.meta[img_name]
        for i in range(len(img_meta)):
            #print("box:", img_meta[i][0], "label", img_meta[i][1])
            boxes.append(img_meta[i][0])
            labels.append(img_meta[i][1])
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
    
    
def load_dataset(meta, root_dir):
    '''Train'''
    #transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = X_View_FasterRCNN(meta=meta, root_dir=root_dir, transforms=transform)
    train_size = int(len(dataset) * 0.8)
    validation_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - validation_size
    # this is a risky operation
    train, validate, test = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])
    #changing batch_size & workers
    loader_train = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True, num_workers=0)
    loader_validate = torch.utils.data.DataLoader(validate, batch_size=1, shuffle=True, num_workers=0)
    loader_test = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True, num_workers=0)

    datasets = {"train": loader_train, "validate": loader_validate, "test": loader_test}
    dataset_lengths = {"train": len(loader_train), "validate": len(loader_validate), "test":len(loader_test)}
    
    return datasets, dataset_lengths