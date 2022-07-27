from .floorplan_WinNet import LoadFloorplan_livwindow,LoadFloorplan_otherwindow
from torch.utils import data
import torch as t
import random
import utils
import os
import numpy as np

class WindowDataset(data.Dataset):
    def __init__(self, data_root, mask_size, mode=0):

        self.mask_size = mask_size
        self.floorplans = [os.path.join(data_root, pth_path) for pth_path in os.listdir(data_root)]
        self.mode=mode

    def __len__(self):
        return len(self.floorplans)

    def __getitem__(self, index):
        floorplan_path = self.floorplans[index]
        if self.mode==0:
            floorplan = LoadFloorplan_livwindow(floorplan_path)
        else:
            floorplan = LoadFloorplan_otherwindow(floorplan_path)
        input=floorplan.get_composite_window()
        target=floorplan.get_targetwindow()
        target=np.uint8(target)
        return input, target