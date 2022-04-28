
from train.WallPlan.data.floorplan_train_LabelNet import LoadFloorplan_LabelNet
import torch as t
import numpy as np
import cv2



def test_Label4GraphNet(file_pth,model,partialGraph,window_mask,all_mask):
    softmax = t.nn.Softmax(dim=1)
    partial_seman_loader = LoadFloorplan_LabelNet(file_pth)
    composite = partial_seman_loader.get_composite_seman()
    t_partialG = t.from_numpy(partialGraph)
    composite[5] = t_partialG
    composite[3] = t.from_numpy(window_mask)
    composite[4] = t.from_numpy(all_mask)
    composite = composite.cuda()
    model.cuda()
    score_model = model(composite.reshape((1, 6, 120, 120)))
    output = np.argmax(softmax(score_model.cpu()).detach().numpy().reshape((8, 120, 120)), axis=0)
    output = np.uint8(output)
    return output