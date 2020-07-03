import time
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import cv2 as cv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import pickle

''' train images klasöründeki resimler pandas dataframe i olarak saklanır. '''
def create_meta(df):
    img_names = df["image_id"]
    img_bound = df["bounds_imcoords"]
    class_names = df["type_id"]
    dir = os.path.join(os.getcwd(), "train_images/")
    df_new = pd.DataFrame(df[["image_id", "bounds_imcoords", "type_id"]])

    return df_new

''' save_obj ve load_obj resim dosyalarını tekrar okumayalım diye pickle dosyası olarak kaydedilir.
'''
def save_obj(obj, name ):
    '''args::kaydedilecek obje:obj, kaydedileceği isim:name'''
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=2)

def load_obj(name):
    '''args aynı dizince olacak dosyanın ismi: name'''
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

'''Her resime ait box koordinatları ve karşılığındaki labellar bir dictionary olarak tutulur.
   Bir resimde birden fazla obje olabilir. . .
'''
def get_masks(path, meta, view_drawings=False):
    curr = None
    boxes = {}
    for index, row in meta.iterrows():
        img_path = os.path.join(path, row[0])
        if curr != row[0]:
            if curr != None and view_drawings is True:
                cv.imshow("img", img1)
                cv.waitKey(0)
            
            img = cv.imread(img_path)
            img1 = np.array(img)
            curr = row[0]
            boxes[curr] = []
        try:
            coord = [int(c) for c in row[1].split(',')] 
            box = img[coord[1]:coord[3], coord[0]:coord[2], :]
            # x0, x1, y0, y1
            #box_cor = [coord[1], coord[3], coord[0], coord[2]]
            box_cor = [coord[1], coord[0], coord[3], coord[2]]
            boxes[curr].append([box_cor, row[2]])
            
            if view_drawings is True:
                color = (255, 0, 0)
                thickness = 2
                cv.rectangle(img1, (coord[0], coord[1]), (coord[2], coord[3]), color, thickness)
            

        except TypeError:
            continue
    return boxes

''' burada eğer dosya önceden kaydedilmişse direkt yüklenir ya da dosya oluşturulur. '''
def system_load(init=False, geojson_file="xView_train.geojson", view_drawings=False):
    '''init should be True if there is not boxes.pkl yet'''
    root_dir = os.path.abspath(os.getcwd())
    if init:
        print("read geojson. . . ")
        df_file = gpd.read_file("xView_train.geojson") # arkadaşın yüklenmesi uzun sürüyor.
        print("completed!")
        print("create csv. . .")
        meta = create_meta(df_file)
        boxes = get_masks(os.path.join(root_dir, "train_images"), meta, view_drawings=view_drawings)
        save_obj(boxes, "boxes")
        print("SUCCESS!")

    else:
        boxes = load_obj("boxes")
        print("meta loaded!")
    
    return (boxes, root_dir)