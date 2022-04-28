
from .floorplan_train_LabelNet import LoadFloorplan_LabelNet
from torch.utils import data
import torch as t
import os
import numpy as np

class LabelNetDataset(data.Dataset):
    def __init__(self, data_root, mask_size):
        self.mask_size = mask_size
        self.floorplans = [os.path.join(data_root, pth_path) for pth_path in os.listdir(data_root) if os.path.splitext(pth_path)[1]=='.pkl']

    def __len__(self):
        return len(self.floorplans)

    def __getitem__(self, index):
        floorplan_path = self.floorplans[index]
        floorplan = LoadFloorplan_LabelNet(floorplan_path, self.mask_size,random_shuffle=True)

        input = floorplan.get_composite_seman()
        target =floorplan.get_target()

        target =np.uint8(target)
        return input, target