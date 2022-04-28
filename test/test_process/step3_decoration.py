import os
import pickle
import copy
import cv2
import numpy as np
import sys
import random
color_map=np.array([
    [244, 241, 222],      #living room
    [234, 182, 159],    #bedroom
    [224, 122, 95],    #kitchen
    [95, 121, 123],    #bathroom
    [242, 204, 143],     #balcony
    [107, 112, 92],        #sotrage
    [100,100,100],      #exterior wall
    [255, 255, 25],     #FrontDoor
    [150,150,150], # interior wall
    [255,255,255]  #external

],dtype=np.int64
)
color_map[:,[0,2]]=color_map[:,[2,0]]

def get_door_win_slices(door_pos_120,gen_wall_graph,room_circles,liv_window_para,other_window_para):
    balcony_win=[]
    no_balcony_win=[]
    boun_slices,boun_searched=find_boundary_slices(gen_wall_graph)
    if len(set(boun_searched))+1!=len(boun_searched):
        print("boundary search repeated")

    frontdoor_door_slice=find_door_slice(door_pos_120,boun_slices,gen_wall_graph)
    if frontdoor_door_slice==None:
        print("not find")

    boun_slices_room_order = []  
    for i in range(len(room_circles)):
        boun_slices_room_order.append([])
    for single_boun_slice in boun_slices:
        room_order=get_boun_slice_order(single_boun_slice, room_circles)
        if room_order!=None:
            single_boun_slice.append(room_circles[room_order]['category'])
            boun_slices_room_order[room_order].append(single_boun_slice)

    balcony_tag=0
    for i in range(len(boun_slices_room_order)):
        if room_circles[i]['category']==4: #balcony
            balcony_tag=1
            single_room_slices=boun_slices_room_order[i]
            for single_slice in single_room_slices:
                for single_win in other_window_para:
                    if judge_pos_in_slice(single_win['pos'],single_slice,gen_wall_graph):
                        single_win_info={}
                        single_win_info['pos']=single_win['pos']
                        single_win_info['ori']=single_win['ori']
                        single_win_info['vertex']=single_slice[1]
                        if single_win_info not in balcony_win:
                            balcony_win.append(single_win_info)
                    else:
                        if single_win not in no_balcony_win:
                            no_balcony_win.append(single_win)
    if balcony_tag==0:
        no_balcony_win.extend(other_window_para)
    all_slices=get_slice_from_circles(room_circles)

    neighbours_slices=get_all_slices_nei(all_slices,room_circles)

    door_slice_room_order = []
    for i in range(len(room_circles)):
        door_slice_room_order.append([])
    for single_slice in neighbours_slices:
        for i in range(len(single_slice['con_tag'])):
            door_slice_room_order[single_slice['con_tag'][i]].append(single_slice)
    return frontdoor_door_slice,boun_slices_room_order,door_slice_room_order,balcony_win,no_balcony_win

def arrange_door_win(gen_wall_graph,room_circles,frontdoor_door_slice,boun_slices_room_order,door_slice_room_order):
    living_ind=0
    for i in range(len(room_circles)):
        if room_circles[i]['category']==0:
            living_ind=i
            break
    setted_front_door=frontdoor_door_slice
    setted_wins=[]
    setted_livwins=[]
    setted_interdoors=[]
    for i in range(len(boun_slices_room_order)):
        room_cate=room_circles[i]['category']
        slices=boun_slices_room_order[i]
        if room_cate==0:
            slices_filter=[slices[ind] for ind in range(len(slices)) if slices[ind][2]>20]
            if len(slices_filter)==1:
                setted_livwins.extend(slices_filter)
            elif len(slices_filter)>1:
                if np.random.rand()>0.5:
                    setted_livwins.extend(random.sample(slices_filter, 2))
                else:
                    setted_livwins.extend(random.sample(slices_filter, 1))
        else:
            slices_filter = [slices[ind] for ind in range(len(slices)) if slices[ind][2] > 10]
            if len(slices_filter)==1:
                setted_wins.extend(slices_filter)
            elif len(slices_filter)>1:
                if np.random.rand()>0.7:
                    setted_wins.extend(random.sample(slices_filter, 2))
                else:
                    setted_wins.extend(random.sample(slices_filter, 1))
    room_marked=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    if len(door_slice_room_order)!=0:
        for door_slice in door_slice_room_order[living_ind]:
            nei_ind=get_nei_ind(door_slice['con_tag'],living_ind)
            if room_marked[nei_ind]==0:
                setted_interdoors.append(door_slice)
                room_marked[nei_ind]=1
    special_balcony_doors=[]

    balcony_inds = []
    for i in range(len(room_circles)):
        if room_circles[i]['category'] == 4:
            balcony_inds.append(i)

    for balcony_ind in balcony_inds:
        balcony_slices=door_slice_room_order[balcony_ind]
        for balcony_slice in balcony_slices:
            slice_len=calculate_slice_len(balcony_slice['vertex'],gen_wall_graph)
            balcony_slice['length']=slice_len
        balcony_slices=sorted(balcony_slices,key=lambda k:(k['length']))
        if len(balcony_slices)>0:
            the_slice=balcony_slices[-1]
            special_balcony_doors.append(the_slice)
    return setted_front_door,setted_interdoors,setted_livwins,setted_wins,special_balcony_doors


def floorplan_render(picture_data_file_pth, gen_wall_graph, room_circles, frontdoor_where, door_where,
                  livwins_where, wins_where, balcony_wins, no_balcony_wins, gen_liv_win, special_balcony_doors, mode=0):

    fp_mask = np.ones((512, 512), dtype=np.uint8)
    fp_mask = fp_mask * 255
    fp_mask = cv2.cvtColor(fp_mask, cv2.COLOR_GRAY2BGR)
    "convert everthing to 512"
    graph_512 = convert_graph_512(gen_wall_graph)
    frontdoor_where_512, door_where_512, livwins_where_512, wins_where_512, balcony_wins_512, no_balcony_wins_512, gen_liv_win_512 = convert_para_512(
        frontdoor_where, door_where, livwins_where, wins_where, balcony_wins, no_balcony_wins, gen_liv_win)

    _, boun_searched = find_boundary_slices(graph_512)

    "Step1. render seman mask"
    render_seman_mask(fp_mask, room_circles, graph_512)

    "Step2. render wall"
    render_wall_mask(fp_mask, graph_512, [150, 150, 150])

    "2.1 render boundary"
    render_boun_mask(fp_mask, graph_512, boun_searched, [100, 100, 100])

    "Step3. render front door"
    render_front_door(fp_mask, frontdoor_where_512)

    "Step4. render inter door"
    render_inter_door(fp_mask, door_where_512,graph_512)

    "Step5. render win"
    render_window(fp_mask, wins_where_512)
    render_liv_window(fp_mask, livwins_where_512)

    return fp_mask

def convert_para_512(frontdoor_where,door_where,livwins_where,wins_where,balcony_wins,no_balcony_wins,gen_liv_win):
    frontdoor_where_512=copy.deepcopy(frontdoor_where)
    door_where_512=copy.deepcopy(door_where)
    livwins_where_512=copy.deepcopy(livwins_where)
    wins_where_512=copy.deepcopy(wins_where)
    gen_liv_win_512=copy.deepcopy(gen_liv_win)

    balcony_wins_512=copy.deepcopy(balcony_wins)
    no_balcony_wins_512=copy.deepcopy(no_balcony_wins)
    for single_frontdoor in frontdoor_where_512:
        single_frontdoor['pos']=np.array(single_frontdoor['pos'])*4
        single_frontdoor['pos'] = np.array(single_frontdoor['pos']) +16

    for single_door in door_where_512:
        single_door['pos']=np.array(single_door['pos'])*4
        single_door['pos'] =np.array(single_door['pos']) +16

    for single_livwin in livwins_where_512:
        single_livwin['pos']=np.array(single_livwin['pos'])*4
        single_livwin['pos'] = np.array(single_livwin['pos']) +16

    for single_win in wins_where_512:
        single_win['pos']=np.array(single_win['pos'])*4
        single_win['pos'] = np.array(single_win['pos']) +16

    for single_balcony in balcony_wins_512:
        single_balcony['pos'] = np.array(single_balcony['pos']) * 4
        single_balcony['pos'] = np.array(single_balcony['pos']) + 16

    for single_no_balcony in no_balcony_wins_512:
        single_no_balcony['pos'][0] = single_no_balcony['pos'][0] * 4
        single_no_balcony['pos'][1] = single_no_balcony['pos'][1] * 4
        single_no_balcony['pos'][0] = single_no_balcony['pos'][0] + 16
        single_no_balcony['pos'][1] = single_no_balcony['pos'][1] + 16

    for single_liv in gen_liv_win_512:
        single_liv['pos'][0] = single_liv['pos'][0] * 4
        single_liv['pos'][1] = single_liv['pos'][1] * 4
        single_liv['pos'][0] = single_liv['pos'][0] + 16
        single_liv['pos'][1] = single_liv['pos'][1] + 16

    return frontdoor_where_512,door_where_512,livwins_where_512,wins_where_512,balcony_wins_512,no_balcony_wins_512,gen_liv_win_512

def render_window(fp_mask,wins_where):
    for single_win in wins_where:
        pos=single_win['pos']
        if single_win['ori']==0:
            fp_mask[pos[0] - 2:pos[0] + 3, pos[1] - 17:pos[1] + 17] = [255, 255, 255]

        else:
            fp_mask[pos[0] - 17:pos[0] + 17, pos[1] - 2:pos[1] + 3] = [255, 255, 255]
def render_liv_window(fp_mask,livwins_where):
    for single_livwin in livwins_where:
        pos=single_livwin['pos']
        if single_livwin['ori']==0:
            fp_mask[pos[0] - 2:pos[0] + 3, pos[1] - 27:pos[1] + 27] = [255, 255, 255]
        else:
            fp_mask[pos[0] - 27:pos[0] + 27, pos[1] - 2:pos[1] + 3] = [255, 255, 255]
def render_inter_door(fp_mask,doors_para,graph_512):
    for single_inter_door in doors_para:

        if single_inter_door['category']!=4:
            pos=single_inter_door['pos']
            if single_inter_door['ori']==0:
                fp_mask[pos[0] - 3:pos[0] + 4, pos[1] - 10:pos[1] + 11] = [255, 255, 255]
            else:
                fp_mask[pos[0] - 10:pos[0] + 11, pos[1] - 3:pos[1] + 4] = [255, 255, 255]
        else:
            v1 = single_inter_door['vertex'][0]
            v2 = single_inter_door['vertex'][1]
            pos=(np.array(graph_512[v1]['pos'])+np.array(graph_512[v2]['pos']))//2
            if single_inter_door['ori']==0:


                door_len=np.abs(graph_512[v1]['pos'][1]-graph_512[v2]['pos'][1])*2//3
                door_len=door_len//2
                fp_mask[pos[0] - 3:pos[0] + 4, pos[1] - door_len:pos[1] + door_len+1] = [255, 255, 255]
            else:

                door_len = np.abs(graph_512[v1]['pos'][0] - graph_512[v2]['pos'][0])*2//3
                door_len = door_len // 2
                fp_mask[pos[0] - door_len:pos[0] + door_len+1, pos[1] - 3:pos[1] +4] = [255, 255, 255]


def render_front_door(fp_mask,front_doors_para):
    for single_front_door in front_doors_para:
        pos=single_front_door['pos']
        if single_front_door['ori']==0:
            fp_mask[pos[0]-3:pos[0]+4,pos[1]-10:pos[1]+10]=color_map[7]
        else:
            fp_mask[pos[0] - 10:pos[0] + 10, pos[1] - 3:pos[1] + 4] = color_map[7]

def render_boun_mask(fp_mask,junction_graph,boun_searched,rgb):
    for ind in range(len(boun_searched)-1):
        pos1=junction_graph[boun_searched[ind]]['pos']
        pos2=junction_graph[boun_searched[ind+1]]['pos']
        fp_mask[pos1[0] - 3:pos1[0] + 4, pos1[1] - 3:pos1[1] + 4] = rgb
        cv2.line(fp_mask,(pos1[1],pos1[0]),(pos2[1],pos2[0]),rgb,5,8)

def render_seman_mask(fp_mask,seman_room_circles,gen_junction_graph):
    for single_seman_room in seman_room_circles:
        circle_inds=single_seman_room['circle']
        points_array = np.array([[gen_junction_graph[ind]['pos'][1],gen_junction_graph[ind]['pos'][0]] for ind in circle_inds] )
        colors=tuple([int(color_map[single_seman_room['category']][i]) for i in range(3)])
        cv2.fillPoly(fp_mask,np.array([points_array]),colors)

def render_wall_mask(fp_mask,graph,color):
    for node in graph:
        if node!=None:
            ori = node['pos']
            fp_mask[ori[0]-3:ori[0]+4,ori[1]-3:ori[1]+4]=color
            for i in node['connect']:
                if i > 0 and graph[i]!=None:
                    target = graph[i]['pos']
                    cv2.line(fp_mask, (ori[1], ori[0]), (target[1], target[0]), color, 5, 4)

def convert_graph_512(graph):
    junction_graph_512 = copy.deepcopy(graph)
    for i in range(len(graph)):
        if junction_graph_512[i] != None:
            junction_graph_512[i]['pos'][0] = junction_graph_512[i]['pos'][0] *4
            junction_graph_512[i]['pos'][1] = junction_graph_512[i]['pos'][1] *4
            junction_graph_512[i]['pos'][0] = junction_graph_512[i]['pos'][0] +16
            junction_graph_512[i]['pos'][1] = junction_graph_512[i]['pos'][1] +16
    return junction_graph_512
def put_door_win(wall_graph,room_circles,front_door_info,front_door_slice,setted_inter_door,setted_livwins,setted_wins):
    front_doors_info=[]
    inter_doors_info=[]
    liv_wins_info=[]
    wins_info=[]

    single_front_door={}
    single_front_door['pos']=front_door_info['pos']
    single_front_door['ori']=front_door_info['ori']
    if front_door_slice!=None:
        if single_front_door['ori']==0:
            single_front_door['pos'][0]=wall_graph[front_door_slice[1][0]]['pos'][0]
        else:
            single_front_door['pos'][1] = wall_graph[front_door_slice[1][0]]['pos'][1]

    front_doors_info.append(single_front_door)

    for single_setted_door in setted_inter_door:
        single_door={}
        inter_door_type=get_door_type(single_setted_door['con_tag'],room_circles)
        door_ori=get_setted_ori(wall_graph,single_setted_door['vertex'])

        door_pos=get_corner_pos(wall_graph,single_setted_door)

        single_door['ori']=door_ori
        single_door['pos']=door_pos
        single_door['category']=inter_door_type
        single_door['vertex']=single_setted_door['vertex']
        inter_doors_info.append(single_door)

    for single_setted_livwin in setted_livwins:
        single_livwin={}
        win_ori=get_setted_ori(wall_graph,single_setted_livwin[1])
        if front_door_slice!=None and  set(single_setted_livwin[1])==set(front_door_slice[1]):
            win_pos=find_extra_win_pos(wall_graph,single_setted_livwin[1],front_door_info['pos'])
        else:
            win_pos=get_setted_avgpos(wall_graph,single_setted_livwin[1])
        single_livwin['ori']=win_ori
        single_livwin['pos']=win_pos
        liv_wins_info.append(single_livwin)

    for single_setted_win in setted_wins:
        single_win={}
        win_ori = get_setted_ori(wall_graph, single_setted_win[1])
        win_type=single_setted_win[3]
        if front_door_slice!=None and  set(single_setted_win[1])==set(front_door_slice[1]):
            win_pos=find_extra_win_pos(wall_graph,single_setted_win[1],front_door_info['pos'])
        else:
            win_pos = get_setted_avgpos(wall_graph, single_setted_win[1])
        single_win['ori']=win_ori
        single_win['pos']=win_pos
        single_win['category']=win_type
        single_win['vertex']=single_setted_win[1]
        wins_info.append(single_win)
    return front_doors_info,inter_doors_info,liv_wins_info,wins_info

def find_extra_win_pos(graph,nei_nodes,door_pos):
    v1_pos=graph[nei_nodes[0]]['pos']
    v2_pos=graph[nei_nodes[1]]['pos']

    door_v1_distance=calculate_distance(door_pos,v1_pos)
    door_v2_distance=calculate_distance(door_pos,v2_pos)

    if door_v1_distance>door_v2_distance:
        return calculate_avg_pos(door_pos,v1_pos)
    else:
        return calculate_avg_pos(door_pos,v2_pos)

def get_setted_avgpos(wall_graph,nei_nodes):
    v1=nei_nodes[0]
    v2=nei_nodes[1]
    return (np.array(wall_graph[v1]['pos'],dtype=np.int32)+np.array(wall_graph[v2]['pos'],dtype=np.int32))//2


def calculate_distance(pos1,pos2):
    pos1=np.array(pos1,dtype=np.int32)
    pos2=np.array(pos2,dtype=np.int32)
    return pow(pos1[0]-pos2[0],2)+pow(pos1[1]-pos2[1],2)
def calculate_avg_pos(pos1,pos2):
    return (np.array(pos1,dtype=np.int32) + np.array(pos2,dtype=np.int32))//2
def get_corner_pos(wall_graph,setted_door):
    junctionG_copy=copy.deepcopy(wall_graph)
    shift_distance = 4
    ind1=setted_door['vertex'][0]
    ind2=setted_door['vertex'][1]
    if np.random.rand()>0.5:
        choice_ind=ind1
        choice_pos=junctionG_copy[choice_ind]['pos']
        target_pos=junctionG_copy[ind2]['pos']
        if target_pos[0]-choice_pos[0]>0:
            choice_pos[0]=choice_pos[0]+shift_distance
        elif target_pos[0]-choice_pos[0]<0:
            choice_pos[0]=choice_pos[0]-shift_distance

        if target_pos[1]-choice_pos[1]>0:
            choice_pos[1] = choice_pos[1] + shift_distance
        elif target_pos[1]-choice_pos[1]<0:
            choice_pos[1] = choice_pos[1] - shift_distance
    else:
        choice_ind=ind2
        choice_pos = junctionG_copy[choice_ind]['pos']
        target_pos = junctionG_copy[ind1]['pos']
        if target_pos[0] - choice_pos[0] > 0:
            choice_pos[0] = choice_pos[0] + shift_distance
        elif target_pos[0] - choice_pos[0] < 0:
            choice_pos[0] = choice_pos[0] - shift_distance

        if target_pos[1] - choice_pos[1] > 0:
            choice_pos[1] = choice_pos[1] + shift_distance
        elif target_pos[1] - choice_pos[1] < 0:
            choice_pos[1] = choice_pos[1] - shift_distance

    return choice_pos

def get_door_type(room_inds,room_circles):
    the_cate=room_circles[room_inds[0]]['category']+room_circles[room_inds[1]]['category']
    return the_cate
def get_setted_ori(graph,nei_nodes):
    v1=nei_nodes[0]
    v2=nei_nodes[1]
    if graph[v1]['pos'][0]==graph[v2]['pos'][0]:
        return 0
    else:
        return 1

def get_nei_ind(list,living_ind):
    for ind in list:
        if ind!=living_ind:
            return ind
    return 0
def calculate_slice_len(vertexs,graph):
    ind1=vertexs[0]
    ind2=vertexs[1]
    length=np.abs(graph[ind1]['pos'][0]-graph[ind2]['pos'][0])+np.abs(graph[ind1]['pos'][1]-graph[ind2]['pos'][1])
    return length

def get_slice_from_circles(room_circles):
    slices=[]
    for room_circle in room_circles:
        circle_inds=room_circle['circle']
        for pos in range(len(circle_inds)-1):
            ind_1=circle_inds[pos]
            ind_2=circle_inds[pos+1]
            single_slice=sorted([ind_1,ind_2])
            if single_slice not in slices:
                slices.append(single_slice)
    return slices

def get_all_slices_nei(all_slices,room_circles):
    neigh_slices=[]
    for single_slice in all_slices:
        room_circle_tag=[]
        for i in range(len(room_circles)):
            circle_inds=room_circles[i]['circle']
            if single_slice[0] in circle_inds and single_slice[1] in circle_inds:
                room_circle_tag.append(i)
        if len(room_circle_tag)>1:
            single_nei_slice={}
            single_nei_slice['vertex']=single_slice
            single_nei_slice['con_tag']=room_circle_tag
            neigh_slices.append(single_nei_slice)
    return neigh_slices

def judge_pos_in_slice(pos,slice,wall_graph):
    ori = slice[0]
    if ori < 2:
        ori = 1
    else:
        ori = 0

def get_boun_slice_order(single_boun_slice,room_circles):
    for i in range(len(room_circles)):
        if single_boun_slice[1][0] in room_circles[i]['circle'] and single_boun_slice[1][1] in room_circles[i]['circle']:
            return i

def find_door_slice(door_pos, boun_slices, wall_graph):
    the_slice = None
    for slice in boun_slices:
        ori = slice[0]
        if ori < 2:
            ori = 1
        else:
            ori = 0
        s_pos = wall_graph[slice[1][0]]['pos']
        e_pos = wall_graph[slice[1][1]]['pos']
        if ori == 0:
            if s_pos[1] < e_pos[1]:
                left = s_pos[1]+2
                right = e_pos[1]-2
            else:
                left = e_pos[1]+2
                right = s_pos[1]-2
            if door_pos[0] in range(s_pos[0] - 3, s_pos[0] + 4) and door_pos[1] in range(left , right + 1 ):
                the_slice = slice
                break
        elif ori == 1:
            if s_pos[0] < e_pos[0]:
                top = s_pos[0]+2
                bottom = e_pos[0]-2
            else:
                top = e_pos[0]+2
                bottom = s_pos[0]-2
            if door_pos[1] in range(s_pos[1] - 3, s_pos[1] + 4) and door_pos[0] in range(top , bottom + 1):
                the_slice = slice
                break
    return the_slice


def find_boundary_slices(boundary_graph):
    start=1
    slices=[]
    searched=[]
    searched.append(start)
    current=start
    next_node=boundary_graph[1]['connect'][3]

    break_times=0
    while(next_node>0):
        break_times=break_times+1
        ori,distance=get_dire_d(boundary_graph,current,next_node)   #current 2 next 的 orientation & distance
        slices.append([ori,[current,next_node],distance])
        searched.append(next_node)
        current=next_node
        next_node=get_next_out(boundary_graph, start, current, ori)

        if break_times>100:
            return slices,searched
    return slices,searched

def get_dire_d(wall_graph, junc1, junc2):
    if wall_graph[junc1]['pos'][0] == wall_graph[junc2]['pos'][0]:
        distance = wall_graph[junc1]['pos'][1] - wall_graph[junc2]['pos'][1]
        if distance > 0:
            return 2, abs(distance)  #
        elif distance < 0:
            return 3, abs(distance)  #
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
        sys.exit(0)
def get_next_out(wall_graph, start_node, current_node, current_ori):
    junction_graph_copy = copy.deepcopy(wall_graph)
    start_pos = junction_graph_copy[start_node]['pos']
    current_pos = junction_graph_copy[current_node]['pos']
    current_con = junction_graph_copy[current_node]['connect']
    current_con[get_reverse_ori(current_ori)] = 0

    "0上 1下 2左 3右"
    "纯右"
    if start_pos[0] == current_pos[0] and start_pos[1] < current_pos[1]:
        if current_ori == 3 or current_ori == 1:
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
    elif start_pos[0] < current_pos[0] and start_pos[1] < current_pos[1]:
        if current_ori == 1 or current_ori == 2:
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
    elif start_pos[0] < current_pos[0] and start_pos[1] == current_pos[1]:
        if current_ori == 2 or current_ori == 0:
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
    elif start_pos[0] < current_pos[0] and start_pos[1] > current_pos[1]:
        if current_ori == 2 or current_ori == 0:
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

def get_room_circles(gen_wall_graph,output_seman,out_seman_8channel):
    gen_circles=get_circles(gen_wall_graph)
    if len(gen_circles)==0:
        return []

    "1.extracted directly from the label map."
    # room_circles = get_room_circles_direct(gen_wall_graph, gen_circles, copy.deepcopy(out_seman_8channel))
    "2.heuristic process ,Living room Kitchen Bathroom first. Recommended"
    room_circles=get_room_circles_heuristic(gen_wall_graph,gen_circles,copy.deepcopy(out_seman_8channel))
    return room_circles

def get_circles(Junction_graph):
    node_prepared = []
    for node in Junction_graph:
        if node != None:
            node_prepared.append([node['index'], node['pos']])
    node_prepared = sorted(node_prepared, key=lambda k: (k[1][0], k[1][1]))
    graph_circles=[]

    while (len(node_prepared)):
        start_node = node_prepared.pop(0)[0]
        if Junction_graph[start_node]['connect'][1] > 0 and Junction_graph[start_node]['connect'][3] > 0:
            current_node = Junction_graph[start_node]['connect'][3]
            current_ori = 3
            one_circle = []
            one_circle.append(start_node)
            one_circle.append(current_node)
            next_node = get_next_node(Junction_graph, start_node, current_node, current_ori)
            break_times = 0
            while (next_node):
                break_times = break_times + 1
                if break_times > 100:
                    break
                if Junction_graph[next_node]['pos'][0] >= Junction_graph[start_node]['pos'][0]:
                    one_circle.append(next_node)
                    current_ori = get_ori_by_points(Junction_graph[current_node]['pos'],
                                                    Junction_graph[next_node]['pos'])
                    if start_node == next_node:
                        break
                    else:
                        current_node = next_node
                else:
                    break
                next_node = get_next_node(Junction_graph, start_node, current_node, current_ori)
            if one_circle[0] == one_circle[-1]:
                graph_circles.append(one_circle)
            else:
                pass
    return graph_circles

def get_room_circles_heuristic(wall_graph, circles, out_seman_8channel):
    room_circles = []
    for i in range(len(circles)):
        room_circles.append(None)
    out_seman_8channel[1] = np.zeros((120, 120))

    "living room, kitchen bathroom, first"
    livingroom_activations = []
    kitchenroom_actications = []
    bathroom_actications = []
    for single_circle in circles:
        circle_detect_mask = np.zeros((120, 120))
        corners_array = [[wall_graph[ind]['pos'][1], wall_graph[ind]['pos'][0]] for ind in single_circle]
        cv2.fillPoly(circle_detect_mask, np.array([corners_array]), 1)
        extract_liv_area = copy.deepcopy(out_seman_8channel[2])
        extract_kitchen_area = copy.deepcopy(out_seman_8channel[4])
        extract_bathroom_area = copy.deepcopy(out_seman_8channel[5])
        extract_liv_area[circle_detect_mask == 0] = 0
        extract_kitchen_area[circle_detect_mask == 0] = 0
        extract_bathroom_area[circle_detect_mask == 0] = 0
        pix_num = circle_detect_mask.sum()
        liv_confidence = extract_liv_area.sum() / pix_num
        kitchen_confidence = extract_kitchen_area.sum() / pix_num
        bathroom_confidence = extract_bathroom_area.sum() / pix_num
        livingroom_activations.append(liv_confidence)
        kitchenroom_actications.append(kitchen_confidence)
        bathroom_actications.append(bathroom_confidence)
    liv_ind = np.argmax(livingroom_activations)  
    kitchenroom_actications[liv_ind] = 0
    bathroom_actications[liv_ind] = 0
    kitch_ind = np.argmax(kitchenroom_actications)
    bathroom_actications[kitch_ind] = 0
    bath_ind = np.argmax(bathroom_actications)
    room_circles[liv_ind] = {}
    room_circles[liv_ind]['circle'] = circles[liv_ind]
    room_circles[liv_ind]['category'] = 0

    room_circles[kitch_ind] = {}
    room_circles[kitch_ind]['circle'] = circles[kitch_ind]
    room_circles[kitch_ind]['category'] = 2

    room_circles[bath_ind] = {}
    room_circles[bath_ind]['circle'] = circles[bath_ind]
    room_circles[bath_ind]['category'] = 3
    out_seman_8channel[2]=np.zeros((120,120))
    out_seman_8channel[4]=np.zeros((120,120))
    seman_mask=np.argmax(out_seman_8channel,0)
    for i in range(len(room_circles)):
        if room_circles[i] == None:
            new_single_circle = {}
            circle_detect_mask = np.zeros((120, 120), dtype=np.uint8)
            corners_array = [[wall_graph[ind]['pos'][1], wall_graph[ind]['pos'][0]] for ind in circles[i]]
            cv2.fillPoly(circle_detect_mask, np.array([corners_array]), 1)
            seman_mask_copy = copy.deepcopy(seman_mask)
            seman_mask_copy[circle_detect_mask == 0] = 0
            category = detect_category(seman_mask_copy)
            new_single_circle['circle'] = circles[i]
            new_single_circle['category'] = category
            room_circles[i] = new_single_circle
    return room_circles


def get_room_circles_direct(wall_graph,circles,seman_mask):
    room_circles=[]
    seman_mask[seman_mask==1]=0
    for single_circle in circles:
        new_single_circle = {}
        circle_detect_mask=np.zeros((120,120),dtype=np.uint8)
        corners_array=[[wall_graph[ind]['pos'][1],wall_graph[ind]['pos'][0]] for ind in single_circle]
        cv2.fillPoly(circle_detect_mask,np.array([corners_array]),1)
        seman_mask_copy=copy.deepcopy(seman_mask)
        seman_mask_copy[circle_detect_mask==0]=0
        category=detect_category(seman_mask_copy)
        new_single_circle['circle']=single_circle
        new_single_circle['category']=category
        room_circles.append(new_single_circle)
    return room_circles

def detect_category(mask):
    flat_mask=mask.flatten()
    counts_mask=np.bincount(flat_mask)
    counts_mask[0]=0
    category=np.argmax(counts_mask)
    if category>=2:
        category=category-2
    return category



def get_ori_by_points(p0,p1):
    if p0[0]==p1[0]:
        if p0[1]<p1[1]:
            return 3
        else:
            return 2
    else:
        if p0[0]<p1[0]:
            return 1
        else:
            return 0

def get_next_node(wall_graph,start_node,current_node,current_ori):
    junction_graph_copy=copy.deepcopy(wall_graph)
    start_pos=junction_graph_copy[start_node]['pos']
    current_pos=junction_graph_copy[current_node]['pos']
    current_con=junction_graph_copy[current_node]['connect']
    current_con[get_reverse_ori(current_ori)]=0
    if start_pos[0]==current_pos[0] and start_pos[1]<current_pos[1]:
        if current_ori==0:
            if current_con[3]:
                return current_con[3]
            elif current_con[0]:
                return current_con[0]
            elif current_con[2]:
                return current_con[2]
            else:
                return 0
        if current_ori==1:
            if current_con[2]:
                return current_con[2]
            elif current_con[1]:
                return current_con[1]
            elif current_con[3]:
                return current_con[3]
            else:
                return 0
        if current_ori==2:
            if current_con[0]:
                return current_con[0]
            elif current_con[2]:
                return current_con[2]
            elif current_con[1]:
                return current_con[1]
            else:
                return 0
        if current_ori==3:
            if current_con[1]:
                return current_con[1]
            elif current_con[3]:
                return current_con[3]
            elif current_con[0]:
                return current_con[0]
            else:
                return 0
    elif start_pos[0]<current_pos[0] and start_pos[1]<current_pos[1]:
        if current_ori==1:
            if current_con[2]:
                return current_con[2]
            elif current_con[1]:
                return current_con[1]
            elif current_con[3]:
                return current_con[3]
            else:
                return 0
        elif current_ori==3:
            if current_con[1]:
                return current_con[1]
            elif current_con[3]:
                return current_con[3]
            elif current_con[0]:
                return current_con[0]
            else:
                return 0
        elif current_ori==0:
            if current_con[3]:
                return current_con[3]
            elif current_con[0]:
                return current_con[0]
            elif current_con[2]:
                return current_con[2]
            else:
                return 0
        elif current_ori==2:
            if current_con[0]:
                return current_con[0]
            elif current_con[2]:
                return current_con[2]
            elif current_con[1]:
                return current_con[1]
            else:
                return 0
    elif start_pos[0]<current_pos[0] and start_pos[1]==current_pos[1]:
        if current_ori==0:
            if current_con[3]:
                return current_con[3]
            elif current_con[0]:
                return current_con[0]
            elif current_con[2]:
                return current_con[2]
            else:
                return 0
        if current_ori==1:
            if current_con[2]:
                return current_con[2]
            elif current_con[1]:
                return current_con[1]
            elif current_con[3]:
                return current_con[3]
            else:
                return 0
        if current_ori==2:
            if current_con[0]:
                return current_con[0]
            elif current_con[2]:
                return current_con[2]
            elif current_con[1]:
                return current_con[1]
            else:
                return 0
        if current_ori==3:
            if current_con[1]:
                return current_con[1]
            elif current_con[3]:
                return current_con[3]
            elif current_con[0]:
                return current_con[0]
            else:
                return 0
    elif start_pos[0]<current_pos[0] and start_pos[1]>current_pos[1]:
        if current_ori==0:
            if current_con[3]:
                return current_con[3]
            elif current_con[0]:
                return current_con[0]
            elif current_con[2]:
                return current_con[2]
            else:
                return 0
        elif current_ori==1:
            if current_con[2]:
                return current_con[2]
            elif current_con[1]:
                return current_con[1]
            elif current_con[3]:
                return current_con[3]
            else:
                return 0
        elif current_ori==2:
            if current_con[0]:
                return current_con[0]
            elif current_con[2]:
                return current_con[2]
            elif current_con[1]:
                return current_con[1]
            else:
                return 0
        elif current_ori==3:
            if current_con[1]:
                return current_con[1]
            elif current_con[3]:
                return current_con[3]
            elif current_con[0]:
                return current_con[0]
            else:
                return 0
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
def clear_graph(junction_graph):
    if get_bump_node(junction_graph):
        Process_bump(junction_graph)

    if get_redundant_node(junction_graph):
        Process_redundant(junction_graph)

    Process_align(junction_graph)
    return junction_graph
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
def Process_bump(graph):
    bump_node_ind=get_bump_node(graph)
    while(bump_node_ind):
        delete_node(graph,bump_node_ind)
        bump_node_ind=get_bump_node(graph)
def delete_node(whole_graph,bump_node_ind):
    whole_graph[bump_node_ind]=None
    for i in range(len(whole_graph)):
        node=whole_graph[i]
        if node!=None:
            for j in range(4):
                if node['connect'][j] and whole_graph[node['connect'][j]]==None:
                    whole_graph[i]['connect'][j]=0
def get_redundant_node(graph):
    for i in range(len(graph)):
        node=graph[i]
        if node!=None and count_list_postive(node['connect'])==2 and ((node['connect'][0] and node['connect'][1]) or (node['connect'][2] and node['connect'][3])):
            ind=node['index']
            return ind
    return 0
def Process_redundant(graph):
    redundant_ind=get_redundant_node(graph)
    while(redundant_ind):
        merge_node(graph,redundant_ind)
        redundant_ind=get_redundant_node(graph)
def merge_node(whole_graph,redundant_ind):
    the_node=whole_graph[redundant_ind]
    the_connect=the_node['connect']
    if the_connect[0] and the_connect[1]:
        whole_graph[the_connect[0]]['connect'][1]=the_connect[1]
        whole_graph[the_connect[1]]['connect'][0]=the_connect[0]
        whole_graph[redundant_ind]=None
    if the_connect[2] and the_connect[3]:
        whole_graph[the_connect[2]]['connect'][3] = the_connect[3]
        whole_graph[the_connect[3]]['connect'][2] = the_connect[2]
        whole_graph[redundant_ind]=None
def Process_align(junction_graph):
    slices_list = get_slices_list(junction_graph)
    for one_slice in slices_list:
        en_align_slice(junction_graph,one_slice)
def get_slices_list(graph):
    slices_list=[]
    for node in graph:
        if node != None:
            h_slice = get_nei_list(graph, node['index'], 0)
            v_slice = get_nei_list(graph, node['index'], 1)
            h_slice_info = {}
            v_slice_info = {}
            h_slice_info['list'] = h_slice
            h_slice_info['ori'] = 0
            v_slice_info['list'] = v_slice
            v_slice_info['ori'] = 1
            slices_list.append(h_slice_info)
            slices_list.append(v_slice_info)
    return slices_list
def get_nei_list(graph, index, ori):
    nei_list = []
    nei_list.append(index)
    if ori == 0:
        temp = graph[index]
        temp = graph[temp['connect'][2]]
        while temp != None:
            nei_list.append(temp['index'])
            temp = graph[temp['connect'][2]]
        temp = graph[index]
        temp = graph[temp['connect'][3]]
        while temp != None:
            nei_list.append(temp['index'])
            temp = graph[temp['connect'][3]]
    elif ori == 1:
        temp = graph[index]
        temp = graph[temp['connect'][0]]
        while temp != None:
            nei_list.append(temp['index'])
            temp = graph[temp['connect'][0]]

        temp = graph[index]
        temp = graph[temp['connect'][1]]

        while temp != None:
            nei_list.append(temp['index'])
            temp = graph[temp['connect'][1]]
    return nei_list
def en_align_slice(graph,one_slice):
    if one_slice['ori']==0:
        avg_h=get_avg_pos(one_slice['list'],graph,0)
        for ind in one_slice['list']:
            graph[ind]['pos'][0]=avg_h
    elif one_slice['ori']==1:
        avg_w=get_avg_pos(one_slice['list'],graph,1)
        for ind in one_slice['list']:
            graph[ind]['pos'][1]=avg_w
def get_avg_pos(list, junction_graph,ori):
    avg_pos=0
    if ori==0:
        for i in range(len(list)):
            avg_pos += junction_graph[list[i]]['pos'][0]

    elif ori==1:
        for i in range(len(list)):
            avg_pos += junction_graph[list[i]]['pos'][1]
    return round(avg_pos/len(list))