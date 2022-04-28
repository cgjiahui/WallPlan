from .floorplan_WinNet import LoadFloorplan_application_livwindow,LoadFloorplan_application_otherwindow
from torch.utils import data
import torch as t
import random
import utils
import os
import numpy as np

class WindowApplicationDataset(data.Dataset):
    def __init__(self, data_root, mask_size, mode=0,constraint_split=[0.1,0.3,0.5]):

        self.mask_size = mask_size
        self.floorplans = [os.path.join(data_root, pth_path) for pth_path in os.listdir(data_root)]
        self.mode=mode
        self.constraint_split=constraint_split

    def __len__(self):
        return len(self.floorplans)

    def __getitem__(self, index):
        floorplan_path = self.floorplans[index]
        if self.mode==0:
            floorplan = LoadFloorplan_application_livwindow(floorplan_path, self.mask_size,constraint_split=self.constraint_split)
        else:
            floorplan = LoadFloorplan_application_otherwindow(floorplan_path, self.mask_size,constraint_split=self.constraint_split)

        input=floorplan.get_composite_window()
        target=floorplan.get_targetwindow()
        target=np.uint8(target)
        return input, target