import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import copy
import pickle
import random
import sys
import re
import numpy as np
import cv2
import utils
from train.WallPlan import models
from test_process.step1_window_assembly import assembling_windows
from test_process.step2_CoupleNet import coupling_networks
from test_process.step3_decoration import clear_graph,get_room_circles,get_door_win_slices,arrange_door_win,put_door_win,floorplan_render

living_window_net = models.model(
    module_name="Living_Win_Net",
    model_name="dlink34no",
    num_classes=2,
    num_channels=4
)
other_window_net = models.model(
    module_name="Other_WinNet",
    model_name="dlink34no",
    num_classes=2,
    num_channels=5
)

label_net = models.model(
        module_name="LabelNet",
        model_name="dlink34no",
        num_classes=8,
        num_channels=6
    )
graph_net = models.model(
        module_name="GraphNet",
        model_name="dlink34no",
        num_classes=3,
        num_channels=8
    )

class WallPlan_Main():
    def __init__(self):
        self.prepare_models()
    def generate_from_val(self,pkl_pth,save_pth):
        with open(pkl_pth, 'rb') as pkl_file:
            [wall_graph,door_info,door_mask_120,boundary_mask_120,boundary_mask_120_5pix,inside_mask_120] = pickle.load(
                pkl_file)
        #For convenience, the input here is redundant, and
        #in fact boundary_mask and inside_maks can be obtained
        #from the wall graph
        plan_name = os.path.split(pkl_pth)[-1]
        print(plan_name)
        junction_graph_120 = convert_graph_120(wall_graph)
        door_info_120 = copy.deepcopy(door_info)
        door_info_120['pos'] = (np.array(door_info['pos']) - 8) // 2
        door_info_120['ori'] = door_info['ori']
        boundary_mask_use = copy.deepcopy(boundary_mask_120)
        inside_mask_use = copy.deepcopy(inside_mask_120)
        door_mask_use = copy.deepcopy(door_mask_120)
        fp_basic = np.zeros((3, 120, 120))
        fp_basic[0] = boundary_mask_use
        fp_basic[1] = inside_mask_use
        fp_basic[2] = door_mask_use

        "Step1. window_mask_generate"
        gen_window_mask,living_windows,other_windows=assembling_windows(boundary_mask_use,inside_mask_use,door_mask_use,living_window_net,other_window_net)

        liv_window_para = []
        other_window_para = []
        for single_liv_window in living_windows:
            liv_para = {}
            pos = [0, 0]
            pos[0] = single_liv_window['para'][0]
            pos[1] = single_liv_window['para'][1]
            ori = single_liv_window['ori']

            liv_para['pos'] = pos
            liv_para['ori'] = ori
            liv_window_para.append(liv_para)
        for single_other_window in other_windows:
            other_para = {}
            if len(single_other_window) >= 4:
                pos = [0, 0]
                pos[0] = single_other_window[0]
                pos[1] = single_other_window[1]
                ori = single_other_window[3]

                other_para['pos'] = pos
                other_para['ori'] = ori
                other_window_para.append(other_para)

        start_pos=junction_graph_120[1]['pos']
        "Step2. graph_generate"
        fp_composite=np.zeros((4,120,120))
        fp_composite[0]=boundary_mask_use
        fp_composite[1]=inside_mask_use
        fp_composite[2]=door_mask_use
        fp_composite[3]=gen_window_mask
        gen_junction_graph,output_seman,output_seman_8channel=coupling_networks(fp_composite,boundary_mask_120_5pix,start_pos,label_net,graph_net)
        gen_junction_graph=clear_graph(gen_junction_graph)
        "2. get room_circles from wall graph"
        room_circles = get_room_circles(gen_junction_graph, output_seman, output_seman_8channel)
        "3. get frontdoor_slice,win_slice_room_order,interdoor_slice_room_order use door_po,wall graph,room_circles"
        door_info_120={}
        door_info_120['pos']=(np.array(door_info['pos'])-8)//2
        door_info_120['ori']=door_info['ori']

        if gen_junction_graph[1]!=None and check_overlap_junction(gen_junction_graph)==0 and check_not_align(gen_junction_graph)==0:
            frontdoor_slice, win_slice_room_order, interdoor_slice_room_order,balcony_wins,no_balcony_wins = get_door_win_slices(
                door_info_120['pos'], gen_junction_graph, room_circles,liv_window_para,other_window_para)

            "4. set setted_front_door,setted_inter_door,setted_livwins,setted_wins"
            setted_front_door, setted_inter_door, setted_livwins, setted_wins,special_balcony_doors = arrange_door_win(
                gen_junction_graph, room_circles, frontdoor_slice, win_slice_room_order, interdoor_slice_room_order)

            "5. put"
            frontdoor_where, door_where, livwins_where, wins_where = put_door_win(gen_junction_graph, room_circles,
                                                                                               door_info_120,
                                                                                               setted_front_door,
                                                                                               setted_inter_door,
                                                                                               setted_livwins, setted_wins)
            "6. rendering "
            fp_mask = floorplan_render(None,gen_junction_graph, room_circles, frontdoor_where, door_where,
                                                     livwins_where, wins_where,balcony_wins,no_balcony_wins,liv_window_para,special_balcony_doors,0)
            cv2.imwrite(f"{save_pth}"+plan_name.replace("pkl","png"),fp_mask)

    def init_input(self,boun_string,frontdoor_string,liv_win_str,win_str):
        boundary_graph=parse_graph(boun_string)
        door_list=parse_frontdoor(frontdoor_string)
        liv_win_list=parse_liv_win(liv_win_str)
        win_list=parse_win(win_str)
        boundary_graph_120=utils.convert_graph_120(boundary_graph)
        door_list_120 = utils.convert_pos_120(door_list)
        liv_win_list_120 = utils.convert_pos_120(liv_win_list)
        win_list_120 = utils.convert_pos_120(win_list)
        return boundary_graph_120,door_list_120,liv_win_list_120,win_list_120

    def prepare_mask(self,boun_G_120,door_list_120,liv_win_list_120,win_list_120):
        boundary_mask_120=utils.restore_from_graph(boun_G_120,120)
        inside_mask_120=utils.restore_inside_mask(boundary_mask_120,boun_G_120)

        boundary_mask_5pix=utils.restore_from_graph_5pix(boun_G_120)

        door_pos=door_list_120[0]['pos']
        front_door_mask_120 = np.zeros((120, 120), np.uint8)
        if door_list_120[0]['ori']==0:
            front_door_mask_120[door_pos[0]-2:door_pos[0]+3,door_pos[1]-3:door_pos[1]+4]=1
        else:
            front_door_mask_120[door_pos[0] - 3:door_pos[0] + 4, door_pos[1] - 2:door_pos[1] + 3] = 1
        window_mask_120=utils.restore_window_mask(liv_win_list_120,win_list_120)

        return boundary_mask_120,boundary_mask_5pix,inside_mask_120,front_door_mask_120,window_mask_120
    def prepare_mask_2(self,boun_G_120,door_list_120):
        boundary_mask_120=utils.restore_from_graph(boun_G_120,120)
        inside_mask_120=utils.restore_inside_mask(boundary_mask_120,boun_G_120)
        boundary_mask_5pix = utils.restore_from_graph_5pix(boun_G_120)
        door_pos=door_list_120[0]['pos']
        front_door_mask_120 = np.zeros((120, 120), np.uint8)
        if door_list_120[0]['ori'] == 0:
            front_door_mask_120[door_pos[0] - 2:door_pos[0] + 3, door_pos[1] - 3:door_pos[1] + 4] = 1
        else:
            front_door_mask_120[door_pos[0] - 3:door_pos[0] + 4, door_pos[1] - 2:door_pos[1] + 3] = 1

        return boundary_mask_120,boundary_mask_5pix,inside_mask_120,front_door_mask_120
    def vis_self_fp(self):
        fp_all_mask=np.copy(self.boun_G_mask*150)
        fp_all_mask[self.front_door_mask>0]=200
        fp_all_mask[self.window_use>0]=250
        cv2.imshow("fp_all_mask",fp_all_mask)
        cv2.waitKey()
    def prepare_models(self):
        living_window_net_pth="../trained_model/Boundary_constraint/WindowLiving.pth"
        other_window_net_pth="../trained_model/Boundary_constraint/WindowOther.pth"
        label_net_pth="../trained_model/Boundary_constraint/LabelNet.pth"
        graph_net_pth="../trained_model/Boundary_constraint/GraphNet.pth"

        living_window_net.load_model(living_window_net_pth)
        other_window_net.load_model(other_window_net_pth)
        label_net.load_model(label_net_pth)
        graph_net.load_model(graph_net_pth)
        living_window_net.cuda()
        other_window_net.cuda()
        label_net.cuda()
        graph_net.cuda()
        living_window_net.eval()
        other_window_net.eval()
        label_net.eval()
        graph_net.eval()

    def predict_windows(self):
        pass

    def predict_start(self):
        pass

    def generate_graph(self):
        pass

    def whole_graphlize(self):
        pass

    def generate_room_circles(self):
        pass

    def arange_win_door(self):
        pass

def save_4_pictures(save_pth,gen_junction_graph,room_circles,balcony_wins,no_balcony_wins,liv_window_para,fp_composite_120):
    pickle_save_file=open(save_pth+"all_composite.pkl","wb")
    pickle.dump([gen_junction_graph,room_circles,balcony_wins,no_balcony_wins,liv_window_para,fp_composite_120],pickle_save_file,protocol=4)
    pickle_save_file.close()
def check_not_align(graph):
    for node in graph:
        if node!=None:
            for con in node['connect']:
                if con:
                    if check_notalign_nodes(graph,node['index'],con):
                        return 1
    return 0



def check_notalign_nodes(graph,ind1,ind2):
    pos1=graph[ind1]['pos']
    pos2=graph[ind2]['pos']
    if pos1[0]!=pos2[0] and pos1[1]!=pos2[1]:
        return 1
    return 0

def check_overlap_junction(whole_graph):
    pos_list=[tuple(node['pos']) for node in whole_graph if node!=None]
    if len(set(pos_list))!=len(pos_list):
        return 1
    else:
        return 0
def convert_graph_512(graph):
    junction_graph_512 = copy.deepcopy(graph)
    for i in range(len(graph)):
        if junction_graph_512[i] != None:
            junction_graph_512[i]['pos'][0] = junction_graph_512[i]['pos'][0] *4
            junction_graph_512[i]['pos'][1] = junction_graph_512[i]['pos'][1] *4
            junction_graph_512[i]['pos'][0] = junction_graph_512[i]['pos'][0] +16
            junction_graph_512[i]['pos'][1] = junction_graph_512[i]['pos'][1] +16
    return junction_graph_512

def convert_graph_120(wall_graph):
    junction_graph_120=copy.deepcopy(wall_graph)
    for i in range(len(junction_graph_120)):
        if junction_graph_120[i]!=None:
            junction_graph_120[i]['pos'][0]=junction_graph_120[i]['pos'][0]-8
            junction_graph_120[i]['pos'][1] = junction_graph_120[i]['pos'][1] - 8
            junction_graph_120[i]['pos'][0] = junction_graph_120[i]['pos'][0]//2
            junction_graph_120[i]['pos'][1] = junction_graph_120[i]['pos'][1]//2
    return junction_graph_120

def convert_graph_512_120(wall_graph):
    junction_graph_120=copy.deepcopy(wall_graph)
    for i in range(len(junction_graph_120)):
        if junction_graph_120[i]!=None:
            junction_graph_120[i]['pos'][0]=junction_graph_120[i]['pos'][0]-16
            junction_graph_120[i]['pos'][1] = junction_graph_120[i]['pos'][1] - 16
            junction_graph_120[i]['pos'][0] = junction_graph_120[i]['pos'][0]//4
            junction_graph_120[i]['pos'][1] = junction_graph_120[i]['pos'][1]//4
    return junction_graph_120

def convert_graph_256_512(wall_graph):
    junction_graph_120=copy.deepcopy(wall_graph)
    for i in range(len(junction_graph_120)):
        if junction_graph_120[i]!=None:
            junction_graph_120[i]['pos'][0] = junction_graph_120[i]['pos'][0]*2
            junction_graph_120[i]['pos'][1] = junction_graph_120[i]['pos'][1]*2
    return junction_graph_120

def get_bump_node(whole_graph):
    for node in whole_graph:
        if node!=None:
            if count_list_postive(node['connect'])==1:
                return node['index']
    return 0
def count_list_postive(list):
    count=0
    for num in list:
        if num>0:
            count=count+1
    return count

def restore_from_graph(graph):
    junction_graph_mask = np.zeros((120, 120), dtype=np.uint8)
    for node in graph:
        if node != None:
            ori = node['pos']
            for i in node['connect']:
                if i > 0:
                    target = graph[i]['pos']
                    cv2.line(junction_graph_mask, (ori[1], ori[0]), (target[1], target[0]), 1, 2, 4)
    return junction_graph_mask

def restore_boundary_120(graph):
    junction_graph_mask = np.zeros((120, 120), dtype=np.uint8)
    for node in graph:
        if node != None:
            ori = node['pos']
            junction_graph_mask[ori[0]-1:ori[0]+2,ori[1]-1:ori[1]+2]=1
            for i in node['connect']:
                if i > 0:
                    target = graph[i]['pos']
                    cv2.line(junction_graph_mask, (ori[1], ori[0]), (target[1], target[0]), 1, 2, 4)
    return junction_graph_mask
def restore_boundary_120_5pix(graph):
    junction_graph_mask = np.zeros((120, 120), dtype=np.uint8)
    for node in graph:
        if node != None:
            ori = node['pos']
            junction_graph_mask[ori[0]-2:ori[0]+3,ori[1]-2:ori[1]+3]=1
            for i in node['connect']:
                if i > 0:
                    target = graph[i]['pos']
                    cv2.line(junction_graph_mask, (ori[1], ori[0]), (target[1], target[0]), 1, 3, 4)

    return junction_graph_mask

def restore_inside_120(graph,boundary_mask):
    inside_mask_120=np.zeros((120,120),dtype=np.uint8)
    _,boun_searched=find_type0_slices(graph)

    points_array = np.array(
        [[graph[ind]['pos'][1], graph[ind]['pos'][0]] for ind in boun_searched])[:-1]
    cv2.fillPoly(inside_mask_120,np.array([points_array]),1)
    inside_mask_120[boundary_mask]=0
    return inside_mask_120


def restore_boun_from_searched(wall_graph,boun_searched):
    boun_5pix=np.zeros((120,120),dtype=np.uint8)

    for ind in range(len(boun_searched)-1):
        pos1=wall_graph[boun_searched[ind]]['pos']
        pos2=wall_graph[boun_searched[ind+1]]['pos']
        boun_5pix[pos1[0]-2:pos1[0]+3,pos1[1]-2:pos1[1]+3]=1
        boun_5pix[pos2[0] - 2:pos2[0] + 3, pos2[1] - 2:pos2[1] + 3] = 1
        cv2.line(boun_5pix,(pos1[1],pos1[0]),(pos2[1],pos2[0]),1,3,4)
    return boun_5pix

def find_type0_slices(wall_graph):
    searched=[]
    slices=[]
    start=1
    searched.append(start)
    current=start
    next_node=wall_graph[1]['connect'][3]
    while(next_node):
        ori,distance=get_dire_d(wall_graph,current,next_node)
        slices.append([ori,[current,next_node],distance])
        searched.append(next_node)
        current=next_node
        next_node=get_next_out(wall_graph,start,current,ori)
    return slices,searched
def get_dire_d(wall_graph, junc1, junc2):
    if wall_graph[junc1]['pos'][0] == wall_graph[junc2]['pos'][0]:
        distance = wall_graph[junc1]['pos'][1] - wall_graph[junc2]['pos'][1]
        if distance > 0:
            return 2, abs(distance)
        elif distance < 0:
            return 3, abs(distance)
        else:
            sys.exit(0)
    elif wall_graph[junc1]['pos'][1] == wall_graph[junc2]['pos'][1]:
        distance = wall_graph[junc1]['pos'][0] - wall_graph[junc2]['pos'][0]
        if distance > 0:
            return 0, abs(distance)
        elif distance < 0:
            return 1, abs(distance)
        else:
            sys.exit(0)
    else:
        return 0
def get_next_out(wall_graph,start_node,current_node,current_ori):
    junction_graph_copy = copy.deepcopy(wall_graph)

    start_pos = junction_graph_copy[start_node]['pos']
    current_pos = junction_graph_copy[current_node]['pos']
    current_con = junction_graph_copy[current_node]['connect']

    current_con[get_reverse_ori(current_ori)] = 0
    if start_pos[0] == current_pos[0] and start_pos[1] < current_pos[1]:
        if current_ori==3 or current_ori==1:
            if current_con[0]:
                return current_con[0]
            elif current_con[3]:
                return current_con[3]
            elif current_con[1]:
                return current_con[1]
            else:
                return current_con[2]
        else:
            if current_con[1]:
                return current_con[1]
            elif current_con[2]:
                return current_con[2]
            elif current_con[0]:
                return current_con[0]
            else:
                return current_con[3]
    elif start_pos[0] <current_pos[0] and start_pos[1]<current_pos[1]:
        if current_ori==1 or current_ori==2:
            if current_con[3]:
                return current_con[3]
            elif current_con[1]:
                return current_con[1]
            elif current_con[2]:
                return current_con[2]
            else:
                return current_con[0]
        else:
            if current_con[2]:
                return current_con[2]
            elif current_con[0]:
                return current_con[0]
            elif current_con[3]:
                return current_con[3]
            else:
                return current_con[1]
    elif start_pos[0]<current_pos[0] and start_pos[1]==current_pos[1]:
        if current_ori==2 or current_ori==0:
            if current_con[1]:
                return current_con[1]
            elif current_con[2]:
                return current_con[2]
            elif current_con[0]:
                return current_con[0]
            else:
                return current_con[3]
        else:
            if current_con[0]:
                return current_con[0]
            elif current_con[3]:
                return current_con[3]
            elif current_con[1]:
                return current_con[1]
            else:
                return current_con[2]
    elif start_pos[0]<current_pos[0] and start_pos[1]>current_pos[1]:
        if current_ori==2 or current_ori==0:
            if current_con[1]:
                return current_con[1]
            elif current_con[2]:
                return current_con[2]
            elif current_con[0]:
                return current_con[0]
            else:
                return current_con[3]
        else:
            if current_con[0]:
                return current_con[0]
            elif current_con[3]:
                return current_con[3]
            elif current_con[1]:
                return current_con[1]
            else:
                return current_con[2]
    else:
        return 0

def get_reverse_ori(ori):
    if ori == 0:
        return 1
    elif ori == 1:
        return 0
    elif ori == 2:
        return 3
    else:
        return 2
def get_door_mask(door_info):
    door_mask=np.zeros((120,120),dtype=np.uint8)
    pos=door_info['pos']
    door_long=7//2
    if door_info['ori']==0:
        door_mask[pos[0]-2:pos[0]+3,pos[1]-door_long:pos[1]+door_long+1]=1
    else:
        door_mask[pos[0]-door_long:pos[0]+door_long+1,pos[1]-2:pos[1]+3]=1
    return door_mask

def parse_graph(string):
    nodes_string = string.split("}")
    boundary_graph=[]
    boundary_graph.append(None)
    for single_node in nodes_string:
        if single_node!='':
            new_node={}
            new_node['connect']=[0,0,0,0]
            new_node['type']=0
            new_node['pos']=[0,0]
            single_node = re.split('[\[\],{} ]', single_node)
            i=0
            for single_value in single_node:
                if single_value != '':
                    if i==0:
                        new_node['index']=np.int32(single_value)
                        i=i+1
                    elif i==1:
                        new_node['connect'][0]=np.int32(single_value)
                        i=i+1
                    elif i==2:
                        new_node['connect'][1]=np.int32(single_value)
                        i=i+1
                    elif i==3:
                        new_node['connect'][2]=np.int32(single_value)
                        i=i+1
                    elif i==4:
                        new_node['connect'][3] = np.int32(single_value)
                        i=i+1
                    elif i==5:
                        new_node['pos'][1] = np.int32(single_value)
                        i=i+1
                    elif i==6:
                        new_node['pos'][0]=np.int32(single_value)
                        i=i+1
            boundary_graph.append(new_node)
    graph_len=len(boundary_graph)
    last_connects=boundary_graph[-1]['connect']
    for i in range(4):
        if last_connects[i]==graph_len:
            boundary_graph[-1]['connect'][i]=1
    return boundary_graph

def parse_frontdoor(frontdoor_string):
    doors=[]
    doors_string=frontdoor_string.split("}")
    for single_door in doors_string:
        if single_door!='':
            new_door={}
            new_door['pos']=[0,0]
            door_para=re.split('[{,\[\] ]',single_door)
            i=0
            for single_value in door_para:
                if single_value!='':
                    if i==0:
                        new_door['ori']=np.int32(single_value)
                        i=i+1
                    elif i==1:
                        new_door['pos'][1]=np.int32(single_value)
                        i=i+1
                    elif i==2:
                        new_door['pos'][0]=np.int32(single_value)
                        i=i+1
            doors.append(new_door)
    return doors

def parse_liv_win(liv_win_str):
    liv_wins=[]
    livwins_string=liv_win_str.split("}")
    for single_livwin in livwins_string:
        if single_livwin!='':
            new_livwin={}
            new_livwin['pos']=[0,0]
            livwin_para=re.split('[{,\[\] ]',single_livwin)
            i=0
            for single_value in livwin_para:
                if single_value!='':
                    if i==0:
                        new_livwin['ori']= np.int32(single_value)
                        i=i+1
                    elif i==1:
                        new_livwin['pos'][1] = np.int32(single_value)
                        i=i+1
                    elif i==2:
                        new_livwin['pos'][0] = np.int32(single_value)
            liv_wins.append(new_livwin)
    return liv_wins

def parse_win(win_str):
    wins=[]
    wins_string=win_str.split("}")
    for single_win in wins_string:
        if single_win!='':
            new_win={}
            new_win['pos']=[0,0]
            win_para=re.split('[{,\[\]]',single_win)
            i=0
            for single_value in win_para:
                if single_value!='':
                    if i==0:
                        new_win['ori']= np.int32(single_value)
                        i=i+1
                    elif i==1:
                        new_win['pos'][1] = np.int32(single_value)
                        i=i+1
                    elif i==2:
                        new_win['pos'][0] = np.int32(single_value)
            wins.append(new_win)
    return wins
def get_samples_from_two(input1,input2,sample_num):
    all_sampled_fp=[]
    all_sampled_fp.append(input1)

    start_boundary_graph=input1['boundary']
    start_door=input1['door']

    end_boundary_graph=input2['boundary']
    end_door=input2['door']

    for i in range(1,sample_num):
        altered_boundary_graph=copy.deepcopy(start_boundary_graph)
        altered_door=copy.deepcopy(start_door)
        altered_door['pos']+=((end_door['pos']-start_door['pos'])*i//sample_num)
        for j in range(len(altered_boundary_graph)):
            if altered_boundary_graph[j]!=None:
                pos_start=start_boundary_graph[j]['pos']
                pos_end  =end_boundary_graph[j]['pos']
                altered_boundary_graph[j]['pos']+=(pos_end-pos_start)*i//sample_num
        all_sampled_fp.append({'boundary':altered_boundary_graph,'door':altered_door})
    all_sampled_fp.append(input2)

    return all_sampled_fp

if __name__=="__main__":
    WallPlan=WallPlan_Main()

    val_pth="./input/"
    val_files = [val_pth + name for name in os.listdir(val_pth) if os.path.splitext(name)[1] == ".pkl"]
    for fp_file in val_files:
        WallPlan.generate_from_val(fp_file,"./output/")



