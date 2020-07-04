import torch
import torchvision
from read_data import load_obj

from engine import evaluate as eval
import utils


def evaluate(model, dataset, device, load_model=True, load_dataset=True, draw_results=False):
    try:
        ''' 
            if load_* params. are True then method accepts path for
            model, dataset, otherwise objects. 
            load_dataset sent without '.pkl'
        '''

        if load_model is True:
            model = torch.load(model) # model is expected to be the model path.
        elif load_dataset is True:
            dataset = load_obj(dataset)
        
        eval(model, dataset, device, draw=draw_results)

    except Exception as e:
        print(e)
        return False
    
    return True
