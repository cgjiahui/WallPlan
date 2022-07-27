from .floorplan_train_GraphNet import LoadFloorplanGraphNet    #注意！！此处加上会和partial_seman训练相冲突！！
from torch.utils import data
import torch as t
import os
import numpy as np
import models
one_model_pth="E:/pycharm_project/Deeplayout/trained_model/doublenet_seman_net/seman_G2P8w_9_9_rightseman_dlink34no_39.pth"
class GraphNetDataset(data.Dataset):
    def __init__(self, data_root, Label_model_pth, mask_size,button=0,process_splits=[0.4,0.5]):
        self.mask_size = mask_size
        self.floorplans = [os.path.join(data_root, pth_path) for pth_path in os.listdir(data_root) if os.path.splitext(pth_path)[1]=='.pkl']
        self.button=button

        model_pth = ""
        self.LabelNet = models.model(
            module_name="LabelNet",
            model_name="dlink34no",
            num_classes=8,
            num_channels=6
        )
        self.LabelNet.load_model(model_pth)
        self.process_splits=process_splits

    def __len__(self):
        return len(self.floorplans)

    def __getitem__(self, index):
        floorplan_path = self.floorplans[index]

        "use the trained model for GraphNet training"
        floorplan = LoadFloorplanGraphNet(floorplan_path,self.LabelNet,self.mask_size,random_shuffle=True,button=self.button,process_splits=self.process_splits)
        input = floorplan.get_composite_junction()
        target =floorplan.get_target_junction()
        target =np.uint8(target)
        target= t.from_numpy(target)
        return input, target

