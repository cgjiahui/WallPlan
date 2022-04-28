import cmath
import copy
import random

import torch as t
from random import sample
import pickle
import utils
import cv2
import os
import numpy as np
pi=3.141592653589793

def get_partial_loading(inter_graph):
    "sample the partial wall"
    condition_mask = np.zeros((120, 120), dtype=np.uint8)
    inter_connections = []
    for node in inter_graph:
        if node != None:
            con_ind1 = node['index']
            for con_ind2 in node['connect']:
                if con_ind2 != 0 and set([con_ind1, con_ind2]) not in inter_connections:
                    inter_connections.append([con_ind1, con_ind2])

    sampled_connections = random.sample(inter_connections, np.random.randint(len(inter_connections) // 5,
                                                                             max(len(inter_connections) // 5 + 1,
                                                                                 len(inter_connections) // 3)))
    for one_connect in sampled_connections:
        pos1 = inter_graph[one_connect[0]]['pos']
        pos2 = inter_graph[one_connect[1]]['pos']
        cv2.line(condition_mask, (pos1[1], pos1[0]), (pos2[1], pos2[0]), 1, 2, 4)
        condition_mask[pos1[0] - 1:pos1[0] + 2, pos1[1] - 1:pos1[1] + 2] = 1
        condition_mask[pos2[0] - 1:pos2[0] + 2, pos2[1] - 1:pos2[1] + 2] = 1

    for node in inter_graph:
        if node != None and np.random.rand() > 0.8:
            pos = node['pos']
            condition_mask[pos[0] - 1:pos[0] + 2, pos[1] - 1:pos[1] + 2] = 2
    return condition_mask
def get_bubble_mask(rooms_info,connects):
    "bubble graph rasterization"
    bubble_node_mask = np.zeros((120, 120), dtype=np.uint8)
    bubble_connect_mask = np.zeros((120, 120), dtype=np.uint8)
    bubble_connect_liv_mask = np.zeros((120, 120), dtype=np.uint8)

    liv_ind=0
    for i in range(len(rooms_info)):
        one_room=rooms_info[i]
        if one_room['category']==0:
            liv_ind=i
        cv2.circle(bubble_node_mask, (one_room['pos'][1], one_room['pos'][0]), area2r(one_room['pixels']) // 3,
                   one_room['category'] + 2, -1)

    for con in connects:
        point1 = rooms_info[con[0]]['pos']
        point2 = rooms_info[con[1]]['pos']
        cv2.line(bubble_connect_mask, (point1[1], point1[0]), (point2[1], point2[0]), 2, 2, 8)
        if liv_ind in con:
            point1 = rooms_info[con[0]]['pos']
            point2 = rooms_info[con[1]]['pos']
            cv2.line(bubble_connect_liv_mask, (point1[1], point1[0]), (point2[1], point2[0]), 1, 2, 8)
            cv2.line(bubble_connect_mask, (point1[1], point1[0]), (point2[1], point2[0]), 1, 2, 8)
    return bubble_node_mask,bubble_connect_mask,bubble_connect_liv_mask

def check_empty(rooms_info):
    for single_room in rooms_info:
        if single_room['pixels']==0:
            return 1
    return 0

def area2r(area):
    return round(np.real(cmath.sqrt(area/pi)))

def list_add(a,b):
    c=[]
    for i in range(len(a)):
        c.append(a[i]+b[i])
    return c



class LoadFloorplan_livwindow():
    def __init__(self, floorplan_graph, mask_size=5, random_shuffle=True):
        with open(floorplan_graph, 'rb') as pkl_file:
            [door_info, boundary_mask_120, inside_mask_120, new_window_mask] = pickle.load(pkl_file)
        plan_name=(os.path.split(floorplan_graph)[-1]).split("multi_")[-1]


        door_info_120=copy.deepcopy(door_info)
        door_info_120['pos'] = (np.array(door_info['pos']) - 8) // 2


        "door_mask"
        door_mask=get_door_mask(door_info_120)
        all_mask=copy.deepcopy(boundary_mask_120)
        all_mask[door_mask>0]=2

        liv_win_mask=np.zeros((120,120),dtype=np.uint8)
        liv_win_mask[new_window_mask== 2]=1


        "input"
        self.boundary_mask=t.from_numpy(boundary_mask_120)
        self.inside_mask=t.from_numpy(inside_mask_120)
        self.front_door_mask=t.from_numpy(door_mask)
        self.all_mask=t.from_numpy(all_mask)


        liv_win_mask=np.zeros((120,120),dtype=np.uint8)
        liv_win_mask[new_window_mask==1]=1

        "output"
        self.gd_liv_window=t.from_numpy(liv_win_mask)

    def get_composite_window(self):
        composite = t.zeros((4, 120, 120))
        composite[0] = self.boundary_mask
        composite[1] = self.inside_mask
        composite[2] = self.front_door_mask
        composite[3] = self.all_mask
        return composite

    def get_targetwindow(self):
        return self.gd_liv_window

class LoadFloorplan_otherwindow():
    def __init__(self, floorplan_graph, mask_size=5, random_shuffle=True):

        with open(floorplan_graph, 'rb') as pkl_file:
            [door_mask_120, boundary_mask_120,inside_mask_120, new_window_mask] = pickle.load(pkl_file)

        liv_win_mask = np.zeros((120, 120), dtype=np.uint8)
        liv_win_mask[new_window_mask == 1] = 1
        door_mask=door_mask_120
        all_mask=copy.deepcopy(boundary_mask_120)
        all_mask[door_mask>0]=2
        all_mask[liv_win_mask>0]=3
        other_win_mask = np.zeros((120, 120), dtype=np.uint8)
        other_win_mask[new_window_mask == 2] = 1
        "input"
        self.boundary_mask=t.from_numpy(boundary_mask_120)
        self.inside_mask=t.from_numpy(inside_mask_120)
        self.front_door_mask=t.from_numpy(door_mask)
        self.liv_win_mask = t.from_numpy(liv_win_mask)
        self.all_mask=t.from_numpy(all_mask)


        "output"
        self.gd_other_window=t.from_numpy(other_win_mask)
    def get_composite_window(self):
        composite = t.zeros((5, 120, 120))
        composite[0] = self.boundary_mask
        composite[1] = self.inside_mask
        composite[2] = self.front_door_mask
        composite[3] = self.liv_win_mask
        composite[4] = self.all_mask
        return composite
    def get_targetwindow(self):
        return self.gd_other_window

class LoadFloorplan_application_livwindow():
    def __init__(self, floorplan_graph, mask_size=5, random_shuffle=True, constraint_split=[0.1,0.3,0.5]):

        with open(floorplan_graph, 'rb') as pkl_file:
            [inter_graph, door_mask_120, boundary_mask_120, inside_mask_120,
             rooms_info, connects, new_window_mask] = pickle.load(pkl_file)

        partial_wall_mask = np.zeros((120, 120), dtype=np.uint8)
        bubble_node_mask = np.zeros((120, 120), dtype=np.uint8)
        bubble_connect_mask = np.zeros((120, 120), dtype=np.uint8)
        bubble_connect_liv_mask = np.zeros((120, 120), dtype=np.uint8)
        random_triger=np.random.rand()
        if random_triger < constraint_split[0]:
            partial_wall_mask = get_partial_loading(inter_graph)
            bubble_node_mask, bubble_connect_mask, bubble_connect_liv_mask = get_bubble_mask(rooms_info, connects)
        elif random_triger < constraint_split[1]:
            bubble_node_mask, bubble_connect_mask, bubble_connect_liv_mask = get_bubble_mask(rooms_info, connects)
        elif random_triger < constraint_split[2]:
            partial_wall_mask = get_partial_loading(inter_graph)
        else:
            pass

        liv_win_mask = np.zeros((120, 120), dtype=np.uint8)
        liv_win_mask[new_window_mask == 1] = 1
        door_mask=door_mask_120
        all_mask=copy.deepcopy(boundary_mask_120)
        all_mask[door_mask>0]=2
        other_win_mask = np.zeros((120, 120), dtype=np.uint8)
        other_win_mask[new_window_mask == 2] = 1
        "input"
        self.boundary_mask=t.from_numpy(boundary_mask_120)
        self.inside_mask=t.from_numpy(inside_mask_120)
        self.front_door_mask=t.from_numpy(door_mask)
        self.all_mask=t.from_numpy(all_mask)
        self.patial_wall_mask=t.from_numpy(partial_wall_mask)
        self.bubble_node_mask=t.from_numpy(bubble_node_mask)
        self.bubble_connect_mask=t.from_numpy(bubble_connect_mask)
        self.bubble_connect_liv_mask=t.from_numpy(bubble_connect_liv_mask)

        "output"
        self.gd_liv_window=t.from_numpy(liv_win_mask)
    def get_composite_window(self):
        composite = t.zeros((8, 120, 120))
        composite[0] = self.boundary_mask
        composite[1] = self.inside_mask
        composite[2] = self.front_door_mask
        composite[3] = self.all_mask
        composite[4] = self.patial_wall_mask
        composite[5] = self.bubble_node_mask
        composite[6] = self.bubble_connect_mask
        composite[7] = self.bubble_connect_liv_mask
        return composite

    def get_targetwindow(self):
        return self.gd_liv_window

class LoadFloorplan_application_otherwindow():
    def __init__(self, floorplan_graph, mask_size=5, random_shuffle=True, constraint_split=[0.1,0.3,0.5]):

        with open(floorplan_graph, 'rb') as pkl_file:
            [ inter_graph, door_mask_120, boundary_mask_120, inside_mask_120,
             rooms_info, connects, new_window_mask] = pickle.load(
                pkl_file)

        partial_wall_mask = np.zeros((120, 120), dtype=np.uint8)
        bubble_node_mask = np.zeros((120, 120), dtype=np.uint8)
        bubble_connect_mask = np.zeros((120, 120), dtype=np.uint8)
        bubble_connect_liv_mask = np.zeros((120, 120), dtype=np.uint8)
        random_triger = np.random.rand()
        if random_triger < constraint_split[0]:
            partial_wall_mask = get_partial_loading(inter_graph)
            bubble_node_mask, bubble_connect_mask, bubble_connect_liv_mask = get_bubble_mask(rooms_info, connects)
        elif random_triger < constraint_split[1]:
            bubble_node_mask, bubble_connect_mask, bubble_connect_liv_mask = get_bubble_mask(rooms_info, connects)
        elif random_triger < constraint_split[2]:
            partial_wall_mask = get_partial_loading(inter_graph)
        else:
            pass

        liv_win_mask = np.zeros((120, 120), dtype=np.uint8)
        liv_win_mask[new_window_mask == 1] = 1
        door_mask = door_mask_120
        all_mask = copy.deepcopy(boundary_mask_120)
        all_mask[door_mask > 0] = 2
        all_mask[liv_win_mask>0]=3
        other_win_mask = np.zeros((120, 120), dtype=np.uint8)
        other_win_mask[new_window_mask == 2] = 1
        "input"
        self.boundary_mask = t.from_numpy(boundary_mask_120)
        self.inside_mask = t.from_numpy(inside_mask_120)
        self.front_door_mask = t.from_numpy(door_mask)
        self.livwindow_mask=t.from_numpy(liv_win_mask)
        self.all_mask = t.from_numpy(all_mask)
        self.patial_wall_mask = t.from_numpy(partial_wall_mask)
        self.bubble_node_mask = t.from_numpy(bubble_node_mask)
        self.bubble_connect_mask = t.from_numpy(bubble_connect_mask)
        self.bubble_connect_liv_mask = t.from_numpy(bubble_connect_liv_mask)

        "output"
        self.gd_other_window = t.from_numpy(other_win_mask)

    def get_composite_window(self):
        composite = t.zeros((9, 120, 120))
        composite[0] = self.boundary_mask
        composite[1] = self.inside_mask
        composite[2] = self.front_door_mask
        composite[3] = self.livwindow_mask
        composite[4] = self.all_mask
        composite[5] = self.patial_wall_mask
        composite[6] = self.bubble_node_mask
        composite[7] = self.bubble_connect_mask
        composite[8] = self.bubble_connect_liv_mask
        return composite

    def get_targetwindow(self):
        return self.gd_other_window

def convert_graph_120(wall_graph):
    junction_graph_120=copy.deepcopy(wall_graph)
    for i in range(len(junction_graph_120)):
        if junction_graph_120[i]!=None:
            junction_graph_120[i]['pos'][0]=junction_graph_120[i]['pos'][0]-8
            junction_graph_120[i]['pos'][1] = junction_graph_120[i]['pos'][1] - 8
            junction_graph_120[i]['pos'][0] = junction_graph_120[i]['pos'][0]//2
            junction_graph_120[i]['pos'][1] = junction_graph_120[i]['pos'][1]//2
    return junction_graph_120
def show_wall(img_path):
    walls=[14,15,16,17]
    print(img_path)
    img=cv2.imread(img_path,-1)
    img_array=np.asarray(img,dtype=np.uint8)
    category_array=img_array[:,:,1]
    cv2.imshow("cate",category_array*100)

    wall_mask=np.zeros(category_array.shape,dtype=np.uint8)

    for wall in walls:
        wall_mask[category_array==wall]=10*wall
        if wall==15 or wall==17:
            wall_mask[category_array==wall]-=100
    cv2.imshow("walls_256",wall_mask)

def get_door_mask(door_info):
    door_mask=np.zeros((120,120),dtype=np.uint8)
    pos=door_info['pos']
    door_long=7//2
    if door_info['ori']==0:
        door_mask[pos[0]-2:pos[0]+3,pos[1]-door_long:pos[1]+door_long+1]=1
    else:
        door_mask[pos[0]-door_long:pos[0]+door_long+1,pos[1]-2:pos[1]+3]=1
    return door_mask
if __name__ == '__main__':
    floorplan_val="G:/sjh/origin_dataset/dataset_8w_11/Junction_graph_8w_split/val/"