import copy
import cmath
import numpy as np
import torch as t
import random
import pickle
import cv2
import os
pi=3.141592653589793

class LoadFloorplan_LabelNet():
    def __init__(self, floorplan_graph, mask_size=5, random_shuffle=True):
        "Read data from pickle"
        with open(floorplan_graph, 'rb') as pkl_file:
            [wall_graph,room_circles,
             door_info, door_mask_120, boundary_mask_120, inside_mask_120,
             new_room_cate_mask, allG_iteration, boun_slices, boun_slices_room_order] = pickle.load(pkl_file)
        door_slice = find_door_slice(door_info['pos'], boun_slices, wall_graph)
        door_info_120 = copy.deepcopy(door_info)
        door_info_120['pos'] = (np.array(door_info['pos']) - 8) // 2
        door_info_120['ori'] = door_info['ori']
        door_info_120['vex'] = door_slice[1]

        junction_graph_120 = convert_graph_120(wall_graph)
        window_mask = get_random_window(junction_graph_120, boun_slices_room_order, room_circles, door_info_120)

        door_mask = door_mask_120
        all_mask = copy.deepcopy(boundary_mask_120)
        all_mask[door_mask > 0] = 2
        all_mask[window_mask == 1] = 3
        all_mask[window_mask == 2] = 4
        _, partial_wall_mask, _ = self.get_train_data_from_graphs(junction_graph_120, allG_iteration)

        "input"
        self.boundary_mask = t.from_numpy(boundary_mask_120)
        self.inside_mask = t.from_numpy(inside_mask_120)
        self.door_mask = t.from_numpy(door_mask)
        self.window_mask = t.from_numpy(window_mask)
        self.all_mask = t.from_numpy(all_mask)

        self.patial_mask = t.from_numpy(partial_wall_mask)

        "output"
        self.gd_seman_mask = t.from_numpy(new_room_cate_mask)

    def get_composite_label(self):
        composite = t.zeros((6, 120, 120))
        composite[0] = self.boundary_mask
        composite[1] = self.inside_mask
        composite[2] = self.door_mask
        composite[3] = self.window_mask
        composite[4] = self.all_mask
        composite[5] = self.patial_mask
        return composite

    def get_target(self):
        return self.gd_seman_mask

    def restore_from_graph(self, graph):
        junction_graph_mask = np.zeros((120, 120), dtype=np.uint8)
        for node in graph:
            if node != None:
                ori = node['pos']
                junction_graph_mask[ori[0] - 2:ori[0] + 3, ori[1] - 2:ori[1] + 3] = 1
                for i in node['connect']:
                    if i > 0:
                        target = graph[i]['pos']
                        cv2.line(junction_graph_mask, (ori[1], ori[0]), (target[1], target[0]), 1, 3, 4)
        return junction_graph_mask

    def get_train_data_from_graphs(self, whole_graph, graphs, button=0):
        G = 1
        input_nodes = []
        output_nodes = []
        last_nodes = []

        in_junction_mask = np.zeros((120, 120))
        in_wall_mask = np.zeros((120, 120))
        groundtruth = np.zeros((120, 120))

        threadshlood = [0.3, 0.5]
        if G == 1:
            L = len(graphs['iteration'])
            if button == 0:
                iter_num = random.randint(1, L)
            elif button == 1:
                triger = np.random.rand()
                if triger < threadshlood[0]:
                    iter_num = random.randint(1, 3)
                else:
                    iter_num = random.randint(1, L)
            else:
                triger = np.random.rand()
                if triger < threadshlood[1]:
                    iter_num = random.randint(1, 3)
                else:
                    iter_num = random.randint(1, L)


            for i in range(iter_num):
                input_nodes.extend(graphs['iteration'][i])
            if iter_num > 0:
                last_nodes.extend(graphs['iteration'][iter_num - 1])

            if iter_num < L:
                output_nodes.extend(graphs['iteration'][iter_num])

        elif G > 1:
            graph_num = random.randint(0, G - 1)
            for i in range(graph_num):
                for nodes in graphs['iteration']:
                    input_nodes.extend(nodes)
            L = len(graphs['iteration'])
            iter_num = random.randint(1, L)

            for i in range(iter_num):
                input_nodes.extend(graphs['iteration'][i])
            if iter_num > 0:
                last_nodes.extend(graphs['iteration'][iter_num - 1])
            elif iter_num == 0 and graph_num > 0:
                last_nodes.extend(graphs['iteration'][-1])

            if iter_num < L:
                output_nodes.extend(graphs['iteration'][iter_num])
        for ind in input_nodes:
            [c_h, c_w] = whole_graph[ind]['pos']
            in_junction_mask[c_h - 2:c_h + 3, c_w - 2:c_w + 3] = 1
            in_wall_mask[c_h - 1:c_h + 2, c_w - 1:c_w + 2] = 1
            for i in whole_graph[ind]['connect']:
                if i > 0 and i in input_nodes:
                    target = whole_graph[i]['pos']
                    cv2.line(in_wall_mask, (c_w, c_h), (target[1], target[0]), 1, 2, 4)

        for ind in last_nodes:
            [c_h, c_w] = whole_graph[ind]['pos']
            in_junction_mask[c_h - 2:c_h + 3, c_w - 2:c_w + 3] = 2.0

        for out_node in output_nodes:
            for last_node in last_nodes:
                if out_node in whole_graph[last_node]['connect']:
                    pos1 = whole_graph[out_node]['pos']
                    pos2 = whole_graph[last_node]['pos']
                    cv2.line(groundtruth, (pos1[1], pos1[0]), (pos2[1], pos2[0]), 1, 3, 4)
            for out_node_in in output_nodes:
                if out_node in whole_graph[out_node_in]['connect']:
                    pos1 = whole_graph[out_node]['pos']
                    pos2 = whole_graph[out_node_in]['pos']
                    cv2.line(groundtruth, (pos1[1], pos1[0]), (pos2[1], pos2[0]), 1, 3, 4)
        for ind in output_nodes:
            [c_h, c_w] = whole_graph[ind]['pos']
            groundtruth[c_h - 2:c_h + 3, c_w - 2:c_w + 3] = 2.0

        return in_junction_mask, in_wall_mask, groundtruth

    def get_plan(self,wall_graph):
        img=np.zeros((120,120),dtype=np.uint8)
        for node in wall_graph:
            if node != None:
                start = node['pos']
                for con in node['connect']:
                    if con > 0:
                        end = wall_graph[con]['pos']
                        cv2.line(img, (start[1], start[0]), (end[1], end[0]), 1, 2, 4)
                img[start[0]-1:start[0]+2,start[1]-1:start[1]+2]=1
        return img


    def get_ground_truth(self,room_circles,wall_graph):
        wall_mask = np.zeros((120, 120), dtype=np.uint8)
        for one_circle in room_circles:
            for i in range(len(one_circle['circle']) - 1):
                start = one_circle['circle'][i]
                next = one_circle['circle'][i + 1]
                pos1 = wall_graph[start]['pos']
                pos2 = wall_graph[next]['pos']
                cv2.line(wall_mask, (pos1[1], pos1[0]), (pos2[1], pos2[0]), 100, 2, 4)
        room_cate_mask = copy.deepcopy(wall_mask)
        for one_circle in room_circles:
            cate = one_circle['category']
            padding_start = [wall_graph[one_circle['circle'][0]]['pos'][0] + 2,
                             wall_graph[one_circle['circle'][0]]['pos'][1] + 2]
            room_cate_mask = self.get_per_room_cate(room_cate_mask, padding_start, cate + 1)
        room_cate_mask[wall_mask > 0] = 0

        return room_cate_mask


    def get_per_room_cate(self,room_cate_mask,padding_start,cate):
        padding_array = []
        room_cate_mask[padding_start[0], padding_start[1]] = cate
        padding_array.append(padding_start)

        while (len(padding_array) > 0):
            pop_pos = padding_array.pop(0)
            # print(pop_pos)
            for i in range(pop_pos[1] - 1, pop_pos[1] + 2):
                for j in range(pop_pos[0] - 1, pop_pos[0] + 2):

                    if room_cate_mask[j, i] == 0:
                        room_cate_mask[j, i] = cate
                        padding_array.append([j, i])

        return room_cate_mask


class LoadFloorplan_application_LabelNet():
    def __init__(self, floorplan_graph, mask_size=5, random_shuffle=True, constraint_split=[0.1,0.3,0.5]):
        "Read data from pickle"

        with open(floorplan_graph,'rb') as pkl_file:
            [wall_graph, inter_graph, room_circles,
             door_info, door_mask_120, boundary_mask_120, inside_mask_120, new_room_cate_mask,
             rooms_info, connects, allG_iteration, boun_slices, boun_slices_room_order] = pickle.load(
                pkl_file)

        "door info"
        door_slice = find_door_slice(door_info['pos'], boun_slices, wall_graph)
        door_info_120 = copy.deepcopy(door_info)
        door_info_120['pos'] = (np.array(door_info['pos']) - 8) // 2
        door_info_120['ori'] = door_info['ori']
        door_info_120['vex'] = door_slice[1]

        junction_graph_120=convert_graph_120(wall_graph)

        "use of randomly distributed windows"
        window_mask=get_random_window(junction_graph_120,boun_slices_room_order,room_circles,door_info_120)

        all_mask=copy.deepcopy(boundary_mask_120)
        all_mask[door_mask_120>0]=2
        all_mask[window_mask==1]=3
        all_mask[window_mask==2]=4
        _,partial_graph_mask,_=self.get_train_data_from_graphs(junction_graph_120,allG_iteration)

        "application input"
        loading_bearing_wall_mask = np.zeros((120, 120), dtype=np.uint8)
        bubble_node_mask = np.zeros((120, 120), dtype=np.uint8)
        bubble_connect_mask = np.zeros((120, 120), dtype=np.uint8)
        bubble_connect_liv_mask = np.zeros((120, 120), dtype=np.uint8)
        random_triger = np.random.rand()
        if random_triger < constraint_split[0]:
            loading_bearing_wall_mask = get_partial_loading(inter_graph)
            bubble_node_mask, bubble_connect_mask, bubble_connect_liv_mask = get_bubble_mask(rooms_info, connects)
        elif random_triger < constraint_split[1]:
            bubble_node_mask, bubble_connect_mask, bubble_connect_liv_mask = get_bubble_mask(rooms_info, connects)
        elif random_triger < constraint_split[2]:
            loading_bearing_wall_mask = get_partial_loading(inter_graph)
        else:
            pass
        "input"
        self.boundary_mask=t.from_numpy(boundary_mask_120)
        self.inside_mask=t.from_numpy(inside_mask_120)
        self.door_mask=t.from_numpy(door_mask_120)
        self.window_mask=t.from_numpy(window_mask)
        self.all_mask=t.from_numpy(all_mask)
        self.patial_graph_mask=t.from_numpy(partial_graph_mask)
        self.bubble_node_mask=t.from_numpy(bubble_node_mask)
        self.bubble_connect_mask=t.from_numpy(bubble_connect_mask)
        self.bubble_connect_liv_mask=t.from_numpy(bubble_connect_liv_mask)
        self.loading_bearing_wall_mask=t.from_numpy(loading_bearing_wall_mask)

        "output"
        self.gd_seman_mask=t.from_numpy(new_room_cate_mask)

    def get_composite_seman(self):
        composite = t.zeros((10, 120, 120))
        composite[0]=self.boundary_mask
        composite[1]=self.inside_mask
        composite[2]=self.door_mask
        composite[3]=self.window_mask
        composite[4]=self.all_mask
        composite[5]=self.patial_graph_mask
        composite[6]=self.bubble_node_mask
        composite[7]=self.bubble_connect_mask
        composite[8]=self.bubble_connect_liv_mask
        composite[9]=self.loading_bearing_wall_mask
        return composite

    def get_target(self):
        return self.gd_seman_mask

    def restore_from_graph(self,graph):
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

    def get_train_data_from_graphs(self,whole_graph,graphs,button=0):
        G=1
        input_nodes = []
        output_nodes = []
        last_nodes=[]

        in_junction_mask=np.zeros((120,120))
        in_wall_mask=np.zeros((120,120))

        groundtruth=np.zeros((120,120))

        threadshlood=[0.3,0.5]


        if G==1:
            L=len(graphs['iteration'])
            if button==0:
                iter_num=random.randint(1,L)
            elif button==1:
                triger=np.random.rand()
                if triger<threadshlood[0]:
                    iter_num=random.randint(1,3)
                else:
                    iter_num = random.randint(1, L)
            else:
                triger=np.random.rand()
                if triger<threadshlood[1]:
                    iter_num=random.randint(1,3)
                else:
                    iter_num=random.randint(1,L)


            for i in range(iter_num):
                input_nodes.extend(graphs['iteration'][i])
            if iter_num>0:
                last_nodes.extend(graphs['iteration'][iter_num-1])

            if iter_num<L:
                output_nodes.extend(graphs['iteration'][iter_num])

        elif G>1:
            graph_num=random.randint(0,G-1)
            for i in range(graph_num):
                for nodes in graphs['iteration']:
                    input_nodes.extend(nodes)

            L=len(graphs['iteration'])
            iter_num=random.randint(1,L)

            for i in range(iter_num):
                input_nodes.extend(graphs['iteration'][i])
            if iter_num>0 :
                last_nodes.extend(graphs['iteration'][iter_num-1])
            elif iter_num==0 and graph_num>0:
                last_nodes.extend(graphs['iteration'][-1])

            if iter_num<L:
                output_nodes.extend(graphs['iteration'][iter_num])

        for ind in input_nodes:
            [c_h,c_w]=whole_graph[ind]['pos']
            in_junction_mask[c_h-2:c_h+3,c_w-2:c_w+3]=1
            in_wall_mask[c_h-1:c_h+2,c_w-1:c_w+2]=1
            for i in whole_graph[ind]['connect']:
                if i>0 and i in input_nodes:
                    target=whole_graph[i]['pos']
                    cv2.line(in_wall_mask,(c_w,c_h),(target[1],target[0]),1,2,4)

        for ind in last_nodes:
            [c_h, c_w] = whole_graph[ind]['pos']
            in_junction_mask[c_h - 2:c_h + 3, c_w - 2:c_w + 3] = 2.0

        for out_node in output_nodes:
            for last_node in last_nodes:
                if out_node in whole_graph[last_node]['connect']:
                    pos1=whole_graph[out_node]['pos']
                    pos2=whole_graph[last_node]['pos']
                    cv2.line(groundtruth,(pos1[1],pos1[0]),(pos2[1],pos2[0]),1,3,4)
            for out_node_in in output_nodes:
                if out_node in whole_graph[out_node_in]['connect']:
                    pos1=whole_graph[out_node]['pos']
                    pos2=whole_graph[out_node_in]['pos']
                    cv2.line(groundtruth,(pos1[1],pos1[0]),(pos2[1],pos2[0]),1,3,4)
        for ind in output_nodes:
            [c_h, c_w] = whole_graph[ind]['pos']
            groundtruth[c_h - 2:c_h + 3, c_w - 2:c_w + 3] = 2.0




        return in_junction_mask,in_wall_mask,groundtruth

    def get_plan(self,wall_graph):
        img=np.zeros((120,120),dtype=np.uint8)
        for node in wall_graph:
            if node != None:
                start = node['pos']
                for con in node['connect']:
                    if con > 0:
                        end = wall_graph[con]['pos']
                        cv2.line(img, (start[1], start[0]), (end[1], end[0]), 1, 2, 4)

                img[start[0]-1:start[0]+2,start[1]-1:start[1]+2]=1
        return img


    def get_ground_truth(self,room_circles,wall_graph):
        wall_mask = np.zeros((120, 120), dtype=np.uint8)
        for one_circle in room_circles:
            for i in range(len(one_circle['circle']) - 1):
                start = one_circle['circle'][i]
                next = one_circle['circle'][i + 1]
                pos1 = wall_graph[start]['pos']
                pos2 = wall_graph[next]['pos']
                cv2.line(wall_mask, (pos1[1], pos1[0]), (pos2[1], pos2[0]), 100, 2, 4)
        room_cate_mask = copy.deepcopy(wall_mask)
        for one_circle in room_circles:
            cate = one_circle['category']
            padding_start = [wall_graph[one_circle['circle'][0]]['pos'][0] + 2,
                             wall_graph[one_circle['circle'][0]]['pos'][1] + 2]
            room_cate_mask = self.get_per_room_cate(room_cate_mask, padding_start, cate + 1)
        room_cate_mask[wall_mask > 0] = 0

        return room_cate_mask


    def get_per_room_cate(self,room_cate_mask,padding_start,cate):
        padding_array = []
        room_cate_mask[padding_start[0], padding_start[1]] = cate
        padding_array.append(padding_start)

        while (len(padding_array) > 0):
            pop_pos = padding_array.pop(0)
            for i in range(pop_pos[1] - 1, pop_pos[1] + 2):
                for j in range(pop_pos[0] - 1, pop_pos[0] + 2):

                    if room_cate_mask[j, i] == 0:
                        room_cate_mask[j, i] = cate
                        padding_array.append([j, i])

        return room_cate_mask

def get_partial_loading(inter_graph):
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

def area2r(area):
    return round(np.real(cmath.sqrt(area/pi)))


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
                left = s_pos[1]
                right = e_pos[1]
            else:
                left = e_pos[1]
                right = s_pos[1]
            if door_pos[0] in range(s_pos[0] - 1, s_pos[0] + 2) and door_pos[1] in range(left, right + 1):
                the_slice = slice
                break
        elif ori == 1:
            if s_pos[0] < e_pos[0]:
                top = s_pos[0]
                bottom = e_pos[0]
            else:
                top = e_pos[0]
                bottom = s_pos[0]
            if door_pos[1] in range(s_pos[1] - 1, s_pos[1] + 2) and door_pos[0] in range(top, bottom + 1):
                the_slice = slice
                break
    return the_slice

def convert_graph_120(wall_graph):
    junction_graph_120=copy.deepcopy(wall_graph)

    for i in range(len(junction_graph_120)):
        if junction_graph_120[i]!=None:
            junction_graph_120[i]['pos'][0]=junction_graph_120[i]['pos'][0]-8
            junction_graph_120[i]['pos'][1] = junction_graph_120[i]['pos'][1] - 8
            junction_graph_120[i]['pos'][0] = junction_graph_120[i]['pos'][0]//2
            junction_graph_120[i]['pos'][1] = junction_graph_120[i]['pos'][1]//2
    return junction_graph_120


def get_random_window(wall_graph,boun_slices_room_order,room_circles,door_info):
    setted_wins=[]
    setted_livwins=[]
    for i in range(len(boun_slices_room_order)):
        room_cate = room_circles[i]['category']
        slices = boun_slices_room_order[i]
        if room_cate == 0:
            slices_filter = [slices[ind] for ind in range(len(slices)) if slices[ind][2] >= 20]

            if len(slices_filter) == 1:
                setted_livwins.extend(slices_filter)
            elif len(slices_filter) > 1:
                slices_filter2 = sorted(slices_filter, key=lambda k: (k[2]))
                setted_livwins.append(slices_filter2[-1])
        else:
            slices_filter = [slices[ind] for ind in range(len(slices)) if slices[ind][2] >= 10]
            if np.random.rand()>0.382:
                if len(slices_filter) == 1:
                    setted_wins.extend(slices_filter)

                elif len(slices_filter) > 1:
                    slices_filter2 = sorted(slices_filter, key=lambda k: (k[2]))
                    setted_wins.append(slices_filter2[-1])
            else:
                if len(slices_filter) == 1:
                    if np.random.rand()>0.15:
                        setted_wins.extend(slices_filter)
                elif len(slices_filter)>1:
                    if np.random.rand()>0.2:
                        setted_wins.extend(random.sample(slices_filter,np.random.randint(1,len(slices_filter)+1)))

    random_win_mask=restore_window_mask(wall_graph,setted_livwins,setted_wins,door_info)
    return random_win_mask

def get_setted_avgpos(wall_graph,nei_nodes):
    v1=nei_nodes[0]
    v2=nei_nodes[1]
    return (np.array(wall_graph[v1]['pos'],dtype=np.int32)+np.array(wall_graph[v2]['pos'],dtype=np.int32))//2

def calculate_avg_pos(pos1,pos2):
    return (np.array(pos1,dtype=np.int32) + np.array(pos2,dtype=np.int32))//2
def calculate_distance(pos1,pos2):
    pos1=np.array(pos1,dtype=np.int32)
    pos2=np.array(pos2,dtype=np.int32)
    return pow(pos1[0]-pos2[0],2)+pow(pos1[1]-pos2[1],2)

def find_extra_win_pos(graph,nei_nodes,door_pos):
    v1_pos=graph[nei_nodes[0]]['pos']
    v2_pos=graph[nei_nodes[1]]['pos']

    door_v1_distance=calculate_distance(door_pos,v1_pos)
    door_v2_distance=calculate_distance(door_pos,v2_pos)

    if door_v1_distance>door_v2_distance:
        return calculate_avg_pos(door_pos,v1_pos)
    else:
        return calculate_avg_pos(door_pos,v2_pos)

def restore_window_mask(wall_graph,setted_livwins,setted_wins,door_info):
    window_mask = np.zeros((120, 120), dtype=np.uint8)

    for i in range(len(setted_wins)):
        single_win_slice=setted_wins[i]
        [ori,[ind1,ind2],_]=single_win_slice
        c_h=(wall_graph[ind1]['pos'][0]+wall_graph[ind2]['pos'][0])//2
        c_w=(wall_graph[ind1]['pos'][1]+wall_graph[ind2]['pos'][1])//2

        window_len=7//2

        if ori==1 or ori==0:
            window_mask[c_h-window_len:c_h+window_len+1,c_w-2:c_w+3]=2
        elif ori==2 or ori==3:
            window_mask[c_h-2:c_h+3,c_w-window_len:c_w+window_len+1]=2

    for i in range(len(setted_livwins)):
        single_livwin=setted_livwins[i]
        [ori, [ind1, ind2], _] = single_livwin
        if set([ind1,ind2])!=set(door_info['vex']):
            livwin_pos=get_setted_avgpos(wall_graph,single_livwin[1])
        else:
            livwin_pos=find_extra_win_pos(wall_graph,single_livwin[1],door_info['pos'])

        window_len=15//2
        [c_h,c_w]=livwin_pos
        if ori==1 or ori==0:
            window_mask[c_h-window_len:c_h+window_len+1,c_w-2:c_w+3]=1
        elif ori==2 or ori==3:
            window_mask[c_h-2:c_h+3,c_w-window_len:c_w+window_len+1]=1
    return window_mask


if __name__ == '__main__':
    train_pth=""
