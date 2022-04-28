import copy
import os
import cv2
import torch as t
import numpy as np

def assembling_windows_bubble(fp_basic,bubble_node_mask,bubble_connect_mask,bubble_connect_liv_mask,model_living,model_other):

    boundary_mask=fp_basic[0]
    inside_mask=fp_basic[1]
    door_mask=fp_basic[2]
    "front_door_mask"
    t_door_mask=t.from_numpy(door_mask)

    "boundary_mask"
    t_boundary_mask=t.from_numpy(boundary_mask)

    "inside_mask"
    t_inside_mask=t.from_numpy(inside_mask)

    "all_mask"
    all_mask1=copy.deepcopy(boundary_mask)
    all_mask1[door_mask>0]=2
    t_all_mask1=t.from_numpy(all_mask1)

    all_mask2 = copy.deepcopy(boundary_mask)
    all_mask2[door_mask > 0] = 2

    softmax = t.nn.Softmax(dim=1)

    composite_living=t.zeros((7,120,120))
    composite_other=t.zeros((8,120,120))
    composite_living[0]=t_boundary_mask
    composite_living[1]=t_inside_mask
    composite_living[2]=t_door_mask
    composite_living[3]=t_all_mask1
    composite_living[4]=t.from_numpy(bubble_node_mask)
    composite_living[5]=t.from_numpy(bubble_connect_mask)
    composite_living[6]=t.from_numpy(bubble_connect_liv_mask)

    composite_other[0]=t_boundary_mask
    composite_other[1]=t_inside_mask
    composite_other[2]=t_door_mask

    composite_living=composite_living.cuda()
    composite_other=composite_other.cuda()

    model_living.cuda()
    model_other.cuda()

    score_living=model_living(composite_living.reshape((1,7,120,120)))
    living_output = np.argmax(softmax(score_living.cpu()).detach().numpy().reshape((2, 120, 120)), axis=0)
    living_output = np.uint8(living_output)
    gen_living_nodes=get_output_nodes(living_output)
    gen_living_para=[]
    for node in gen_living_nodes:
        pixel_num=len(node)
        c_h,c_w=get_avg_pos(node)
        gen_living_para.append([c_h,c_w,pixel_num])
    nodes_pixels=[gen_living_para[i][2] for i in range(len(gen_living_para))]

    living_windows=[]

    if len(nodes_pixels)!=0:
        living_ind=np.argmax(nodes_pixels)
        living_window={}
        living_window['para']=gen_living_para[living_ind]
        len_h,len_w=get_ori_len(gen_living_nodes,living_ind)
        living_window['ori']=np.uint8(len_w>len_h)
        living_win_restore=restore_window(living_window,0)
        living_windows.append(living_window)
    else:
        living_win_restore=np.zeros((120,120),dtype=np.uint8)

    re_gen_output=copy.deepcopy(living_win_restore)

    re_gen_living_win=copy.deepcopy(np.uint8(living_win_restore>0))
    t_re_living_win=t.from_numpy(re_gen_living_win)
    composite_other[3]=t_re_living_win
    all_mask2[re_gen_living_win>0]=3
    t_all_mask2=t.from_numpy(all_mask2)
    composite_other[4]=t_all_mask2
    composite_other[5]=t.from_numpy(bubble_node_mask)
    composite_other[6]=t.from_numpy(bubble_connect_mask)
    composite_other[7]=t.from_numpy(bubble_connect_liv_mask)

    score_other=model_other(composite_other.reshape((1,8,120,120)))
    other_output=np.argmax(softmax(score_other.cpu()).detach().numpy().reshape((2, 120, 120)), axis=0)
    other_output=np.uint8(other_output)
    gen_other_nodes=get_output_nodes(other_output)

    gen_other_nodes_filter1=[]
    for node in gen_other_nodes:
        if len(node)>(5*6+2):
            gen_other_nodes_filter1.append(node)

    gen_other_para=[]
    for node in gen_other_nodes_filter1:
        pixel_num=len(node)
        c_h,c_w=get_avg_pos(node)
        gen_other_para.append([c_h,c_w,pixel_num])

    re_gen_other_mask=np.zeros((120,120),dtype=np.uint8)
    for i in range(len(gen_other_para)):
        ori=0
        len_h,len_w=get_ori_len(gen_other_nodes_filter1,i)
        if 5 in [len_h,len_w]:
            [c_h,c_w,_]=gen_other_para[i]
            if len_w>len_h:
                ori=1
            if ori==0:
                re_gen_other_mask[c_h-3:c_h+3+1,c_w-2:c_w+3]=1
            else:
                re_gen_other_mask[c_h-2:c_h+3,c_w-3:c_w+3+1]=1
            gen_other_para[i].append(ori)

    re_gen_output[re_gen_other_mask>0]=2
    re_gen_output=np.uint8(re_gen_output)
    return re_gen_output,living_windows,gen_other_para

def assembling_windows(boundary_mask,inside_mask,door_mask,model_living,model_other):
    "front_door_mask"
    t_door_mask=t.from_numpy(door_mask)
    "boundary_mask"
    t_boundary_mask=t.from_numpy(boundary_mask)
    "inside_mask"
    t_inside_mask=t.from_numpy(inside_mask)
    "all_mask"
    all_mask1=copy.deepcopy(boundary_mask)
    all_mask1[door_mask>0]=2
    t_all_mask1=t.from_numpy(all_mask1)

    all_mask2 = copy.deepcopy(boundary_mask)
    all_mask2[door_mask > 0] = 2

    softmax = t.nn.Softmax(dim=1)

    composite_living=t.zeros((4,120,120))
    composite_other=t.zeros((5,120,120))
    composite_living[0]=t_boundary_mask
    composite_living[1]=t_inside_mask
    composite_living[2]=t_door_mask
    composite_living[3]=t_all_mask1

    composite_other[0]=t_boundary_mask
    composite_other[1]=t_inside_mask
    composite_other[2]=t_door_mask


    composite_living=composite_living.cuda()
    composite_other=composite_other.cuda()

    "generate living room window"
    score_living=model_living(composite_living.reshape((1,4,120,120)))
    living_output = np.argmax(softmax(score_living.cpu()).detach().numpy().reshape((2, 120, 120)), axis=0)
    living_output = np.uint8(living_output)

    "standardlize"
    gen_living_nodes=get_output_nodes(living_output)
    gen_living_para=[]
    for node in gen_living_nodes:
        pixel_num=len(node)
        c_h,c_w=get_avg_pos(node)
        gen_living_para.append([c_h,c_w,pixel_num])
    nodes_pixels=[gen_living_para[i][2] for i in range(len(gen_living_para))]

    living_windows=[]

    if len(nodes_pixels)!=0:
        living_ind=np.argmax(nodes_pixels)
        living_window={}
        living_window['para']=gen_living_para[living_ind]
        len_h,len_w=get_ori_len(gen_living_nodes,living_ind)
        living_window['ori']=np.uint8(len_w>len_h)
        living_win_restore=restore_window(living_window,0)
        living_windows.append(living_window)
    else:
        living_win_restore=np.zeros((120,120),dtype=np.uint8)

    "generated_living_window"
    re_gen_output=copy.deepcopy(living_win_restore)
    "step2. generate other window"
    re_gen_living_win=copy.deepcopy(np.uint8(living_win_restore>0))
    t_re_living_win=t.from_numpy(re_gen_living_win)

    "prepare input 4 other_window net"
    composite_other[3]=t_re_living_win
    all_mask2[re_gen_living_win>0]=3
    t_all_mask2=t.from_numpy(all_mask2)
    composite_other[4]=t_all_mask2

    score_other=model_other(composite_other.reshape((1,5,120,120)))
    other_output=np.argmax(softmax(score_other.cpu()).detach().numpy().reshape((2, 120, 120)), axis=0)
    other_output=np.uint8(other_output)

    gen_other_nodes=get_output_nodes(other_output)

    gen_other_nodes_filter1=[]
    for node in gen_other_nodes:
        if len(node)>(5*6+2):
            gen_other_nodes_filter1.append(node)

    gen_other_para=[]
    for node in gen_other_nodes_filter1:
        pixel_num=len(node)
        c_h,c_w=get_avg_pos(node)
        gen_other_para.append([c_h,c_w,pixel_num])

    re_gen_other_mask=np.zeros((120,120),dtype=np.uint8)
    for i in range(len(gen_other_para)):
        ori=0
        len_h,len_w=get_ori_len(gen_other_nodes_filter1,i)
        if 5 in [len_h,len_w]:
            [c_h,c_w,_]=gen_other_para[i]
            if len_w>len_h:
                ori=1
            if ori==0:
                re_gen_other_mask[c_h-3:c_h+3+1,c_w-2:c_w+3]=1
            else:
                re_gen_other_mask[c_h-2:c_h+3,c_w-3:c_w+3+1]=1
            gen_other_para[i].append(ori)

    re_gen_output[re_gen_other_mask>0]=2
    re_gen_output=np.uint8(re_gen_output)
    return re_gen_output,living_windows,gen_other_para

def restore_window(windows,mode):
    the_mask=np.zeros((120,120))
    if mode==0:
        len=15//2
        [c_h,c_w,num]=windows['para']
        ori=windows['ori']
        if ori==1:
            the_mask[c_h-2:c_h+3,c_w-len:c_w+len+1]=1
        else:
            the_mask[c_h-len:c_h+len+1,c_w-2:c_w+3]=1
    return the_mask

def get_output_nodes(output_copy):
    generated_nodes=[]

    copy2=np.zeros((120,120),dtype=np.uint8)

    copy2[output_copy>0]=1
    for x in range(120):
        for y in range(120):
            if copy2[x, y] != 0:
                node = Extract_one_junction(copy2, x, y)
                generated_nodes.append(node)
    return generated_nodes

def Extract_one_junction(gen_copy,x,y):
    node=[]
    node_array=[]
    node_array.append([x,y])
    gen_copy[x, y] = 0

    while(len(node_array)!=0):
        point=node_array.pop(0)
        node.append(point)

        for pos_x in range(point[0]-1,point[0]+2):
            for pos_y in range(point[1]-1,point[1]+2):
                if pos_x in range(120) and pos_y in range(120):
                    if(gen_copy[pos_x,pos_y]!=0):
                        node_array.append([pos_x,pos_y])
                        gen_copy[pos_x,pos_y]=0
    return node

def get_avg_pos(node):
    pixel_num=len(node)
    node_array=np.array(node)

    [avg_h,avg_w]=np.sum(node_array,axis=0)
    avg_h=np.int32(round(avg_h/pixel_num))
    avg_w=np.int32(round(avg_w/pixel_num))
    return avg_h,avg_w

def get_ori_len(gen_win_nodes,i):
    single_nodes=gen_win_nodes[i]
    nodes_sort_h=sorted(single_nodes,key=lambda k:(k[0]))
    h_min=nodes_sort_h[0][0]
    h_max=nodes_sort_h[-1][0]
    nodes_sort_w=sorted(single_nodes,key=lambda k:(k[1]))
    w_min=nodes_sort_w[0][1]
    w_max=nodes_sort_w[-1][1]
    return (h_max-h_min+1),(w_max-w_min+1)