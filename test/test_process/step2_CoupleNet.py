import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import copy
import sys
import cv2
import torch as t
import numpy as np

def test_graph_generate_app_new8w(fp_basic,fp_condi,gen_window_mask,boundary_mask_5pix,start_pos,seman_model,bfs_model):
    softmax = t.nn.Softmax(dim=1)
    window_mask=gen_window_mask

    "1.初始化，获得各项信息"
    "带mask为tensor 不带为numpy   默认全为浮点（显示时转为uint8"
    boundary_mask = fp_basic[0]
    inside_mask = fp_basic[1]
    frontdoor_mask = fp_basic[2]
    all_mask = copy.deepcopy(boundary_mask)
    all_mask[frontdoor_mask > 0] = 2
    all_mask[window_mask == 1] = 3
    all_mask[window_mask == 2] = 4

    bubble_node_mask=fp_condi[0]
    bubble_connect_mask=fp_condi[1]
    bubble_connect_liv_mask=fp_condi[2]



    # cv2.imshow("getted_win", np.uint8(window_mask) * 100)

    t_boundary_mask = t.from_numpy(boundary_mask)
    t_inside_mask = t.from_numpy(inside_mask)
    t_frontdoor_mask = t.from_numpy(frontdoor_mask)
    t_window_mask = t.from_numpy(window_mask)
    t_all_mask = t.from_numpy(all_mask)
    t_bubble_node_mask=t.from_numpy(bubble_node_mask)
    t_bubble_connect_mask=t.from_numpy(bubble_connect_mask)
    t_bubble_connect_liv_mask=t.from_numpy(bubble_connect_liv_mask)

    "inter_mask用于过滤  使用boundary graph恢复 bounmask，结合Inside mask用于过滤"
    "用boundary graph获得 boun_5pix"

    "用来过滤的 boun3_pix为不生成墙遮罩    用boun_5pix生成所有内部遮罩"
    inter_mask = copy.deepcopy(inside_mask)
    inter_mask[boundary_mask_5pix > 0] = 1  # 查看一下用于过滤的inter mask

    "in_junction mask"
    in_junction = np.zeros((120, 120))
    "in_wall mask"
    in_wall = np.zeros((120, 120))
    in_wall_3pix = np.zeros((120, 120))

    # cv2.imshow("win_mask",window_mask*100)
    # cv2.waitKey()
    "初始化 junction_graph信息  iters_nodes信息"
    wall_graph = []
    wall_graph.append(None)
    new_node = {}
    new_node['index'] = 1
    # print(graphs)
    # pos = junction_graph_120[graphs[0]['start']]['pos']  # 嵌套循环这个pos位置
    pos = start_pos
    # star_pos_remask=copy.deepcopy(boundary_mask_real*100)
    # star_pos_remask[pos[0]-2:pos[0]+3,pos[1]-2:pos[1]+3]=200
    # cv2.imshow("start_pos_mask",star_pos_remask)

    new_node['pos'] = [pos[0], pos[1]]
    # print(f"新加的节点是:{new_node['pos']}")
    new_node['connect'] = [0, 0, 0, 0]
    new_node['type'] = 0
    wall_graph.append(new_node)

    iters_nodes = []  # 按迭代顺序加入到Junction graph中的节点
    iters_nodes.append([1])
    last_nodes = iters_nodes[-1]

    # show_junc_graph(whole_graph,"whole_groundtruth",boundary)   #查看一下ground truth
    # seman inwall为3pixel   bfs inwall为5pixel
    "in_junction,in_wall 通过iters_nodes/junction graph获得，每次输入网络训练前更新"
    for nodes in iters_nodes:
        for node in nodes:
            [c_h, c_w] = wall_graph[node]['pos']
            in_junction[c_h - 2:c_h + 3, c_w - 2:c_w + 3] = 1
            in_wall_3pix[c_h - 1:c_h + 2, c_w - 1:c_w + 2] = 1
            # 初始化时没有inwall，故inwall可以不补方形
            for i in wall_graph[node]['connect']:  # 注意是在junction graph查找
                if i > 0:
                    target = wall_graph[i]['pos']
                    cv2.line(in_wall, (c_w, c_h), (target[1], target[0]), 1, 3, 4)
                    cv2.line(in_wall_3pix, (c_w, c_h), (target[1], target[0]), 1, 2, 4)

    for node in last_nodes:
        [c_h, c_w] = wall_graph[node]['pos']
        in_junction[c_h - 2:c_h + 3, c_w - 2:c_w + 3] = 2
    in_junction_mask = t.from_numpy(in_junction)
    # in_bfs_wall_mask = t.from_numpy(in_wall)

    t_in_wall_3pix = t.from_numpy(in_wall_3pix)
    "in partial plan"
    # in_partial_plan = copy.deepcopy(t_boundary_mask.numpy())
    # in_partial_plan[in_wall_3pix > 0] = 1
    # in_partial_plan = t.from_numpy(in_partial_plan)

    "out seman mask"
    seman_composite = t.zeros((9, 120, 120))
    seman_composite[0] = t_boundary_mask
    seman_composite[1] = t_inside_mask
    seman_composite[2] = t_frontdoor_mask
    seman_composite[3] = t_window_mask
    seman_composite[4] = t_all_mask
    seman_composite[5] = t_in_wall_3pix
    seman_composite[6] = t_bubble_node_mask
    seman_composite[7] = t_bubble_connect_mask
    seman_composite[8] = t_bubble_connect_liv_mask

    seman_model.cuda()
    seman_composite = seman_composite.cuda()
    score_seman = seman_model(seman_composite.reshape(1, 9, 120, 120))
    output_seman_8channel = softmax(score_seman.cpu()).detach().numpy().reshape((8, 120, 120))
    output_seman = np.argmax(output_seman_8channel, axis=0)
    # cv2.imshow("out_put_seman1", np.uint8(output_seman) * 33)
    # storage_mask2=np.zeros((120,120),dtype=np.uint8)
    # storage_mask2[output_seman==7]=255
    # cv2.imshow("storage2",storage_mask2)

    t_out_seman = t.from_numpy(output_seman)

    "带mask为tensor，输入到网络进行预测"
    bfs_composite = t.zeros((11, 120, 120))
    bfs_composite[0] = t_boundary_mask
    bfs_composite[1] = t_inside_mask
    bfs_composite[2] = t_frontdoor_mask
    bfs_composite[3] = t_out_seman
    bfs_composite[4] = t_window_mask
    bfs_composite[5] = t_all_mask
    bfs_composite[6] = in_junction_mask
    bfs_composite[7] = t_in_wall_3pix

    bfs_composite[8] = t_bubble_node_mask
    bfs_composite[9] = t_bubble_connect_mask
    bfs_composite[10] = t_bubble_connect_liv_mask

    "展示 input"
    in_img = copy.deepcopy(np.uint8(t_boundary_mask.numpy()) * 100)
    in_img[t_frontdoor_mask > 0] = 255
    in_img[t_window_mask.numpy() > 0] = 200
    in_img[in_junction_mask > 0] = 50
    cv2.imshow("init_input", in_img)

    pure_in_img = copy.deepcopy(np.uint8(t_boundary_mask.numpy()) * 100)
    pure_in_img[t_frontdoor_mask > 0] = 255
    cv2.imshow("pure_input", pure_in_img)

    bfs_model = bfs_model.cuda()
    bfs_composite = bfs_composite.cuda()

    score_model = bfs_model(bfs_composite.reshape((1, 11, 120, 120)))
    output = np.argmax(softmax(score_model.cpu()).detach().numpy().reshape((3, 120, 120)), axis=0)
    output = np.uint8(output)

    "更换过滤策略"
    out_filter = copy.deepcopy(output)
    out_filter[inter_mask == 0] = 0
    output = out_filter

    "output:generated junction mask   iters_nodes:迭代的nodes组  wall_graph:已加入node的graph"
    output_copy = np.copy(output)
    generated_nodes = get_output_nodes(output_copy)
    # print(len(generated_nodes[0]))
    # print(generated_nodes)

    "2.处理.Filter 简单处理过滤，先设置<18pixel不要" "动态阈值获得多样性结果？"
    gen_nodes_filter1 = []
    for node in generated_nodes:
        if len(node) > 7:  # 3pix 版本，阈值设为7?
            gen_nodes_filter1.append(node)

    "最终将生成的junction 转为gen_nodes_para，并用来 更新junction graph和iters 存储[c_h,c_w,pix_num]"
    gen_nodes_para = []
    for node in gen_nodes_filter1:  # 顺序性应和filter2相同
        pixel_num = len(node)
        center_h, center_w = get_avg_pos(node)
        gen_nodes_para.append([center_h, center_w, pixel_num])

    break_count = 0

    "3.开始迭代    过程:确定Filter后的节点和Last_nodes 已经之间的连接，添加，更新junction_graph iters"
    # 1.将gen_nodes_filter1,确定连接后加进graph/iters 2.更新输入的mask，再得到output  3.更新得到gen_nodes_filter1
    # 更新iters_nodes wall_graph   iters_nodes[-1]是last_nodes
    # gen_nodes_para[]
    dai = 0
    while (len(gen_nodes_filter1) > 0):
        break_count = break_count + 1
        if break_count > 50:
            break

        dai = dai + 1
        # print(f"diedai:{dai}")
        new_iter = []  # new_iter是本次加入的节点，如果为0死循环情况！！如果为0，写一个动态调整判断junction阈值
        last_nodes = iters_nodes[-1]

        "重要！ gen_nodes_para"

        "Step1.使用新的思路确定上次生成junction和已有之间的连接，"
        "新思路：从每个last_node上下左右对齐方向找并排序判断 "
        "原則：从上下左右找最近连接"
        "同方向的连完"

        "判断一下已加入graph的还是否有连接"

        "写一个，同方向选择最优情况连"
        "1.连接last node和本次节点之间的连接"
        # print(f"生成了{len(gen_nodes_para)}个点:{gen_nodes_para}")
        for last_node in last_nodes:
            last_pos = wall_graph[last_node]['pos']
            for i in range(4):  # i代表上下左右方向
                if wall_graph[last_node]['connect'][i] == 0:
                    neighbors = get_neighbors(last_pos, gen_nodes_para, i)
                    # print(f"给点{last_pos} 找{i}方向，找到{len(neighbors)}个邻居")
                    # print(f"neighbors 为:{neighbors}")   #neighbors 参数:[c_h,c_w,pix_num]
                    sorted_nei = []
                    "选近的"
                    if i == 0 or i == 1:
                        sorted_nei = sorted(neighbors, key=lambda k: (k[0]))
                    else:
                        sorted_nei = sorted(neighbors, key=lambda k: (k[1]))
                    "选多的"
                    # sorted_nei = sorted(neighbors, key=lambda k: (k[2]), reverse=True)
                    if len(sorted_nei) > 0:
                        "选近的"
                        if i == 0 or i == 2:
                            selected_node = sorted_nei[-1]  # 方向相关
                        else:
                            selected_node = sorted_nei[0]

                        "选多的"
                        # selected_node = sorted_nei[0]
                        "根据两点位置之间的output像素密度决定是否连接"  # 加一个判断条件，两位置之间没有加好的junction
                        if judge_connect(last_pos, [selected_node[0], selected_node[1]], output) and no_inter_junction(
                                last_pos, [selected_node[0], selected_node[1]], wall_graph):
                            ori = i
                            # print(f"diedai:{dai},pos1:f{last_pos},pos2:{[gen_node[0], gen_node[1]]}")
                            apended_pos = [wall_graph[i]['pos'] for i in new_iter]
                            if [selected_node[0], selected_node[1]] not in apended_pos:
                                new_node = {}
                                new_node['index'] = wall_graph[-1]['index'] + 1
                                new_node['pos'] = [selected_node[0], selected_node[1]]

                                new_ori = get_reverse_ori(ori)
                                connect = [0, 0, 0, 0]
                                connect[new_ori] = last_node
                                new_node['connect'] = connect
                                wall_graph.append(new_node)
                                wall_graph[last_node]['connect'][ori] = new_node['index']

                                new_iter.append(new_node['index'])

                            else:
                                index = new_iter[-1]
                                for ind in new_iter:
                                    if [selected_node[0], selected_node[1]] == wall_graph[ind]['pos']:
                                        index = ind  # 内部的index，作用域在下面不可见，必须在上循环外定义
                                        break
                                wall_graph[last_node]['connect'][ori] = index
                                wall_graph[index]['connect'][get_reverse_ori(ori)] = last_node

        candidate_nodes = get_candidate_nodes(wall_graph, np.uint8(output > 0))
        for choice_ind in range(len(candidate_nodes) - 1):
            for compare_ind in range(choice_ind + 1, len(candidate_nodes)):
                ind1 = candidate_nodes[choice_ind]
                ind2 = candidate_nodes[compare_ind]
                node1 = wall_graph[ind1]
                node2 = wall_graph[ind2]
                if may_connect(node1['pos'], node2['pos']):
                    if no_inter_junction(node1['pos'], node2['pos'], wall_graph) and judge_connect_3pix(
                            node1['pos'], node2['pos'], output):  # 更严格的测试，使用5pix的测试,阈值0.75
                        target_ori = get_target_ori(node1['pos'], node2['pos'])
                        reverse_ori = get_reverse_ori(target_ori)

                        if node1['connect'][target_ori] == 0 and node2['connect'][reverse_ori] == 0:
                            wall_graph[ind1]['connect'][target_ori] = ind2
                            wall_graph[ind2]['connect'][reverse_ori] = ind1

        if len(new_iter) > 1:
            new_nodes_para = []
            for ind in new_iter:
                pos = wall_graph[ind]['pos']
                for node_para in gen_nodes_para:
                    if pos[0] == node_para[0] and pos[1] == node_para[1]:
                        new_nodes_para.append(node_para)
            for new_node in new_iter:
                new_pos = wall_graph[new_node]['pos']
                for i in range(4):
                    if wall_graph[new_node]['connect'][i] == 0:
                        neighbors = get_neighbors(new_pos, new_nodes_para, i)

                        if i == 0 or i == 1:
                            sorted_nei = sorted(neighbors, key=lambda k: (k[0]))
                        else:
                            sorted_nei = sorted(neighbors, key=lambda k: (k[1]))

                        if len(sorted_nei) > 0:
                            if i == 0 or i == 2:
                                selected_node = sorted_nei[-1]
                            else:
                                selected_node = sorted_nei[0]

                            for iter_node in wall_graph:
                                if iter_node != None:
                                    if iter_node['pos'][0] == selected_node[0] and iter_node['pos'][1] == \
                                            selected_node[1]:
                                        selected_ind = iter_node['index']
                                        select_pos = iter_node['pos']

                            if no_inter_junction(new_pos, select_pos, wall_graph) and \
                                    wall_graph[selected_ind]['connect'][
                                        get_reverse_ori(i)] == 0 and judge_connect(new_pos, select_pos, output):
                                wall_graph[new_node]['connect'][i] = selected_ind
                                wall_graph[selected_ind]['connect'][get_reverse_ori(i)] = new_node

        iters_nodes.append(new_iter)
        last_nodes = iters_nodes[-1]
        if len(new_iter) == 0:
            break
        in_junction = np.zeros((120, 120))
        in_wall = np.zeros((120, 120))
        in_wall_3pix = np.zeros((120, 120))
        for nodes in iters_nodes:
            for node in nodes:
                [c_h, c_w] = wall_graph[node]['pos']
                in_junction[c_h - 2:c_h + 3, c_w - 2:c_w + 3] = 1
                in_wall_3pix[c_h - 1:c_h + 2, c_w - 1:c_w + 2] = 1
                in_wall[c_h - 2:c_h + 3, c_w - 2:c_w + 3] = 1
                for i in wall_graph[node]['connect']:  # 注意是在junction graph查找
                    if i > 0:
                        target = wall_graph[i]['pos']
                        cv2.line(in_wall, (c_w, c_h), (target[1], target[0]), 1, 3, 4)  # 数据格式和训练时保持一致
                        cv2.line(in_wall_3pix, (c_w, c_h), (target[1], target[0]), 1, 2, 4)
        for node in last_nodes:
            [c_h, c_w] = wall_graph[node]['pos']
            in_junction[c_h - 2:c_h + 3, c_w - 2:c_w + 3] = 2

        "in jun/wall for bfs"
        in_junction_mask = t.from_numpy(in_junction)

        "in partial plan for seman"
        t_in_wall_3pix = t.from_numpy(in_wall_3pix)

        "new out seman mask"
        seman_composite = seman_composite.reshape((9, 120, 120))
        seman_composite[5] = t_in_wall_3pix
        score_seman = seman_model(seman_composite.reshape((1, 9, 120, 120)))
        output_seman_8channel = softmax(score_seman.cpu()).detach().numpy().reshape((8, 120, 120))
        output_seman = np.argmax(output_seman_8channel, axis=0)


        t_out_seman = t.from_numpy(output_seman)

        bfs_composite = bfs_composite.reshape((11, 120, 120))
        bfs_composite[3] = t_out_seman
        bfs_composite[6] = in_junction_mask
        bfs_composite[7] = t_in_wall_3pix

        score_model = bfs_model(bfs_composite.reshape((1, 11, 120, 120)))
        output = np.argmax(softmax(score_model.cpu()).detach().numpy().reshape((3, 120, 120)), axis=0)
        output = np.uint8(output)  # output 是本次生成的
        out_filter = copy.deepcopy(output)
        out_filter[inter_mask == 0] = 0
        output = out_filter

        after = np.uint8(in_wall_3pix) * 100
        after[output > 0.5] = 120
        after[output > 1.5] = 255
        after[in_junction > 0] = 100
        after[in_junction > 1.5] = 150
        output_copy = np.copy(output)
        generated_nodes = get_output_nodes(output_copy)
        gen_nodes_filter1 = []
        for node in generated_nodes:
            if len(node) > 7:
                gen_nodes_filter1.append(node)
        gen_nodes_para = []
        for node in gen_nodes_filter1:
            pixel_num = len(node)
            center_h, center_w = get_avg_pos(node)
            gen_nodes_para.append([center_h, center_w, pixel_num])

        gen_nodes_restore = np.zeros((120, 120), dtype=np.uint8)
        for node in gen_nodes_para:
            [center_h, center_w, _] = node
            gen_nodes_restore[center_h - 2:center_h + 3, center_w - 2:center_w + 3] = 200
    return wall_graph, output_seman, output_seman_8channel

def coupling_networks(fp_composite,boundary_mask_5pix,start_pos,Label_Net,Graph_Net):
    softmax = t.nn.Softmax(dim=1)

    "0. Prepare inputs"
    boundary_mask=fp_composite[0]
    inside_mask=fp_composite[1]
    frontdoor_mask=fp_composite[2]
    window_mask=fp_composite[3]

    all_mask=copy.deepcopy(boundary_mask)
    all_mask[frontdoor_mask>0]=2
    all_mask[window_mask==1]=3
    all_mask[window_mask==2]=4

    t_boundary_mask=t.from_numpy(boundary_mask)
    t_inside_mask=t.from_numpy(inside_mask)
    t_frontdoor_mask=t.from_numpy(frontdoor_mask)
    t_window_mask=t.from_numpy(window_mask)
    t_all_mask=t.from_numpy(all_mask)

    inter_mask=copy.deepcopy(inside_mask)
    inter_mask[boundary_mask_5pix>0]=1 

    in_junction = np.zeros((120, 120))
    in_wall_3pix = np.zeros((120, 120))
    
    "init generated wall graph"
    wall_graph = []
    wall_graph.append(None)
    new_node = {}
    new_node['index'] = 1

    pos = start_pos
    
    new_node['pos'] = [pos[0], pos[1]]
    new_node['connect'] = [0, 0, 0, 0]
    new_node['type'] = 0
    wall_graph.append(new_node)
    
    "Generated nodes in one iteration"
    iters_nodes = []  
    iters_nodes.append([1])
    last_nodes = iters_nodes[-1]

    for nodes in iters_nodes:
        for node in nodes:
            [c_h, c_w] = wall_graph[node]['pos']
            in_junction[c_h - 2:c_h + 3, c_w - 2:c_w + 3] = 1
            in_wall_3pix[c_h - 1:c_h + 2, c_w - 1:c_w + 2] = 1
            for i in wall_graph[node]['connect']:
                if i > 0:
                    target = wall_graph[i]['pos']
                    cv2.line(in_wall_3pix, (c_w, c_h), (target[1], target[0]), 1, 2, 4)

    for node in last_nodes:
        [c_h, c_w] = wall_graph[node]['pos']
        in_junction[c_h - 2:c_h + 3, c_w - 2:c_w + 3] = 2
    in_junction_mask = t.from_numpy(in_junction)
    t_in_wall_3pix=t.from_numpy(in_wall_3pix)

    "the output label mask (8 channel and 1 channel"
    label_composite = t.zeros((6, 120, 120))
    label_composite[0] = t_boundary_mask
    label_composite[1] = t_inside_mask
    label_composite[2] = t_frontdoor_mask
    label_composite[3] = t_window_mask
    label_composite[4] = t_all_mask
    label_composite[5] = t_in_wall_3pix

    label_composite = label_composite.cuda()
    score_label = Label_Net(label_composite.reshape(1, 6, 120, 120))
    output_label_8channel = softmax(score_label.cpu()).detach().numpy().reshape((8, 120, 120))
    output_label = np.argmax(output_label_8channel, axis=0)

    t_out_label = t.from_numpy(output_label)

    "the output graph(wall) mask"
    graph_composite = t.zeros((8, 120, 120))
    graph_composite[0] = t_boundary_mask
    graph_composite[1] = t_inside_mask
    graph_composite[2] = t_frontdoor_mask
    graph_composite[3] = t_out_label
    graph_composite[4] = t_window_mask
    graph_composite[5] = t_all_mask
    graph_composite[6] = in_junction_mask
    graph_composite[7] = t_in_wall_3pix

    Graph_Net = Graph_Net.cuda()
    graph_composite = graph_composite.cuda()

    score_model = Graph_Net(graph_composite.reshape((1, 8, 120, 120)))
    output = np.argmax(softmax(score_model.cpu()).detach().numpy().reshape((3, 120, 120)), axis=0)
    output = np.uint8(output)

    out_filter=copy.deepcopy(output)
    out_filter[inter_mask==0]=0
    output=out_filter

    output_copy = np.copy(output)
    generated_nodes = get_output_nodes(output_copy)
    gen_nodes_filter1 = []
    for node in generated_nodes:
        if len(node) > 7:
            gen_nodes_filter1.append(node)
    gen_nodes_para = []
    for node in gen_nodes_filter1:
        pixel_num = len(node)
        center_h, center_w = get_avg_pos(node)
        gen_nodes_para.append([center_h, center_w, pixel_num])

    break_count=0
    dai = 0
    while (len(gen_nodes_filter1) > 0):
        break_count=break_count+1
        if break_count>50:
            break

        dai = dai + 1
        new_iter = []
        last_nodes = iters_nodes[-1]

        "Find the newly generated nodes from nodes joined in previous round"
        for last_node in last_nodes:
            last_pos = wall_graph[last_node]['pos']
            for i in range(4):
                if wall_graph[last_node]['connect'][i] == 0:
                    neighbors = get_neighbors(last_pos, gen_nodes_para, i)

                    "Sort by the distance"
                    if i == 0 or i == 1:
                        sorted_nei = sorted(neighbors, key=lambda k: (k[0]))
                    else:
                        sorted_nei = sorted(neighbors, key=lambda k: (k[1]))
                    if len(sorted_nei) > 0:
                        if i == 0 or i == 2:
                            selected_node = sorted_nei[-1]
                        else:
                            selected_node = sorted_nei[0]

                        "judge the connection"
                        if judge_connect(last_pos, [selected_node[0], selected_node[1]], output) and no_inter_junction(last_pos,[selected_node[0], selected_node[1]],wall_graph):
                            ori = i
                            apended_pos = [wall_graph[i]['pos'] for i in new_iter]
                            if [selected_node[0], selected_node[1]] not in apended_pos:
                                new_node = {}
                                new_node['index'] = wall_graph[-1]['index'] + 1
                                new_node['pos'] = [selected_node[0], selected_node[1]]

                                new_ori = get_reverse_ori(ori)
                                connect = [0, 0, 0, 0]
                                connect[new_ori] = last_node
                                new_node['connect'] = connect
                                wall_graph.append(new_node)
                                wall_graph[last_node]['connect'][ori] = new_node['index']
                                new_iter.append(new_node['index'])

                            else:
                                index = new_iter[-1]
                                for ind in new_iter:
                                    if [selected_node[0], selected_node[1]] == wall_graph[ind]['pos']:
                                        index = ind
                                        break
                                wall_graph[last_node]['connect'][ori] = index
                                wall_graph[index]['connect'][get_reverse_ori(ori)] = last_node

        possible_nodes = get_possible_nodes(wall_graph,np.uint8(output>0))

        "other possible connections"
        for choice_ind in range(len(possible_nodes)-1):
            for compare_ind in range(choice_ind+1,len(possible_nodes)):
                ind1=possible_nodes[choice_ind]
                ind2=possible_nodes[compare_ind]
                node1=wall_graph[ind1]
                node2=wall_graph[ind2]

                if may_connect(node1['pos'],node2['pos']):
                    "Determination of connection possibilities"
                    if no_inter_junction(node1['pos'],node2['pos'],wall_graph) and  judge_connect_3pix(node1['pos'],node2['pos'],output):
                        target_ori=get_target_ori(node1['pos'],node2['pos'])
                        reverse_ori=get_reverse_ori(target_ori)
                        if node1['connect'][target_ori]==0 and node2['connect'][reverse_ori]==0:
                            wall_graph[ind1]['connect'][target_ori]=ind2
                            wall_graph[ind2]['connect'][reverse_ori]=ind1

        "Find the connection between the nodes generated in this round"
        if len(new_iter) > 1:
            new_nodes_para = []
            for ind in new_iter:
                pos = wall_graph[ind]['pos']
                for node_para in gen_nodes_para:
                    if pos[0] == node_para[0] and pos[1] == node_para[1]:
                        new_nodes_para.append(node_para)
            for new_node in new_iter:
                new_pos = wall_graph[new_node]['pos']
                for i in range(4):
                    if wall_graph[new_node]['connect'][i] == 0:
                        neighbors = get_neighbors(new_pos, new_nodes_para, i)
                        if i == 0 or i == 1:
                            sorted_nei = sorted(neighbors, key=lambda k: (k[0]))
                        else:
                            sorted_nei = sorted(neighbors, key=lambda k: (k[1]))

                        if len(sorted_nei) > 0:
                            if i == 0 or i == 2:
                                selected_node = sorted_nei[-1]
                            else:
                                selected_node = sorted_nei[0]

                            for iter_node in wall_graph:
                                if iter_node != None:
                                    if iter_node['pos'][0] == selected_node[0] and iter_node['pos'][1] == \
                                            selected_node[1]:
                                        selected_ind = iter_node['index']
                                        select_pos = iter_node['pos']
                            if no_inter_junction(new_pos, select_pos,wall_graph) and wall_graph[selected_ind]['connect'][
                                get_reverse_ori(i)] == 0 and judge_connect(new_pos, select_pos, output):
                                wall_graph[new_node]['connect'][i] = selected_ind
                                wall_graph[selected_ind]['connect'][get_reverse_ori(i)] = new_node

        iters_nodes.append(new_iter)
        last_nodes = iters_nodes[-1]
        if len(new_iter) == 0:
            break

        "Prepare for the next round prediction"
        in_junction = np.zeros((120, 120))
        in_wall = np.zeros((120, 120))
        in_wall_3pix = np.zeros((120, 120))
        for nodes in iters_nodes:
            for node in nodes:
                [c_h, c_w] = wall_graph[node]['pos']
                in_junction[c_h - 2:c_h + 3, c_w - 2:c_w + 3] = 1
                in_wall_3pix[c_h - 1:c_h + 2, c_w - 1:c_w + 2] = 1
                in_wall[c_h - 2:c_h + 3, c_w - 2:c_w + 3] = 1
                for i in wall_graph[node]['connect']:
                    if i > 0:
                        target = wall_graph[i]['pos']
                        cv2.line(in_wall, (c_w, c_h), (target[1], target[0]), 1, 3, 4)
                        cv2.line(in_wall_3pix, (c_w, c_h), (target[1], target[0]), 1, 2, 4)
        for node in last_nodes:
            [c_h, c_w] = wall_graph[node]['pos']
            in_junction[c_h - 2:c_h + 3, c_w - 2:c_w + 3] = 2

        "in jun/wall for bfs"
        in_junction_mask = t.from_numpy(in_junction)

        "in partial plan for label"
        t_in_wall_3pix=t.from_numpy(in_wall_3pix)

        "new out label mask"
        label_composite = label_composite.reshape((6, 120, 120))
        label_composite[5] = t_in_wall_3pix
        score_label = Label_Net(label_composite.reshape((1, 6, 120, 120)))
        output_label_8channel = softmax(score_label.cpu()).detach().numpy().reshape((8, 120, 120))
        output_label = np.argmax(output_label_8channel, axis=0)

        t_out_label = t.from_numpy(output_label)

        graph_composite = graph_composite.reshape((8, 120, 120))
        graph_composite[3] = t_out_label
        graph_composite[6] = in_junction_mask
        graph_composite[7] = t_in_wall_3pix

        score_model = Graph_Net(graph_composite.reshape((1, 8, 120, 120)))
        output = np.argmax(softmax(score_model.cpu()).detach().numpy().reshape((3, 120, 120)), axis=0)
        output = np.uint8(output)
        out_filter = copy.deepcopy(output)
        out_filter[inter_mask == 0] = 0
        output = out_filter
        output_copy = np.copy(output)
        generated_nodes = get_output_nodes(output_copy)
        gen_nodes_filter1 = []
        for node in generated_nodes:
            if len(node) > 7:
                gen_nodes_filter1.append(node)

        gen_nodes_para = []
        for node in gen_nodes_filter1:
            pixel_num = len(node)
            center_h, center_w = get_avg_pos(node)
            gen_nodes_para.append([center_h, center_w, pixel_num])

        gen_nodes_restore = np.zeros((120, 120), dtype=np.uint8)
        for node in gen_nodes_para:
            [center_h, center_w, _] = node
            gen_nodes_restore[center_h - 2:center_h + 3, center_w - 2:center_w + 3] = 200
    return wall_graph,output_label,output_label_8channel

def judge_connect_5pix(pos1, pos2, output, radio=0.7):
    output_copy = np.copy(output)

    test_mask = np.zeros((120, 120), dtype=np.uint8)
    cv2.line(test_mask, (pos1[1], pos1[0]), (pos2[1], pos2[0]), 1, 3, 4)
    output_copy = np.uint8(output_copy > 0)
    if np.sum(output_copy & test_mask) / np.sum(test_mask) > 0.75:
        return 1
    else:
        return 0

def restore_from_graph(graph):
    junction_graph_mask = np.zeros((120, 120), dtype=np.uint8)
    for node in graph:
        if node != None:
            ori = node['pos']
            for i in node['connect']:
                if i > 0:
                    target = graph[i]['pos']
                    cv2.line(junction_graph_mask, (ori[1], ori[0]), (target[1], target[0]), 1, 2, 4)
    for node in graph:
        if node != None:
            ori = node['pos']
            junction_graph_mask[ori[0]-1:ori[0]+2,ori[1]-1:ori[1]+2]=2
    return junction_graph_mask

def get_target_ori(pos1,pos2):
    pos1=np.int32(pos1)
    pos2=np.int32(pos2)

    if abs(pos1[1]-pos2[1]) < abs(pos1[0]-pos2[0]):
        if pos1[0]<pos2[0]:
            return 1
        else:
            return 0
    else:
        if pos1[1]<pos2[1]:
            return 3
        else:
            return 2

def judge_connect_3pix(pos1, pos2, output, radio=0.7):
    output_copy = np.copy(output)

    test_mask = np.zeros((120, 120), dtype=np.uint8)
    cv2.line(test_mask, (pos1[1], pos1[0]), (pos2[1], pos2[0]), 1, 2, 4)
    output_copy = np.uint8(output_copy > 0)
    if np.sum(output_copy & test_mask) / np.sum(test_mask) > 0.75:
        return 1
    else:
        return 0

def may_connect(pos1,pos2):
    return min( abs(np.int32(pos1[0])-np.int32(pos2[0])),abs(np.int32(pos1[1])-np.int32(pos2[1])))<=2

def count_junction_area(pos,mask):
    [h,w]=pos
    count=0
    for i in range(max(h-1,0),min(h+2,119)):
        for j in range(max(w-1,0),min(w+2,119)):
            if mask[i,j]>0:
                count=count+1

    return count

def get_possible_nodes(grpah,output):
    candidates_nodes=[]
    for node in grpah:
        if node!=None:
            if count_junction_area(node['pos'],output)>7:
                candidates_nodes.append(node['index'])
    return candidates_nodes

def get_reverse_ori(ori):
    if ori == 0:
        return 1
    elif ori == 1:
        return 0
    elif ori == 2:
        return 3
    else:
        return 2

def no_inter_junction(pos1,pos2,wall_graph):
    flag=1
    test_mask=np.zeros((120,120),dtype=np.uint8)
    cv2.line(test_mask,(pos1[1],pos1[0]),(pos2[1],pos2[0]),1,2,4)
    test_mask[pos1[0]-2:pos1[0]+3,pos1[1]-2:pos1[1]+3]=0
    test_mask[pos2[0] - 2:pos2[0] + 3, pos2[1] - 2:pos2[1] + 3] = 0

    for node in wall_graph:
        if node!=None:
            pos=node['pos']
            if test_mask[pos[0],pos[1]]==1:
                return 0
    return 1

def judge_connect(pos1, pos2, output, radio=0.7):
    output_copy = np.copy(output)

    test_mask = np.zeros((120, 120), dtype=np.uint8)
    cv2.line(test_mask, (pos1[1], pos1[0]), (pos2[1], pos2[0]), 1, 2, 4)
    output_copy = np.uint8(output_copy > 0)
    if np.sum(output_copy & test_mask) / np.sum(test_mask) > 0.7:
        return 1
    else:
        return 0

def get_neighbors(ori_pos, gen_nodes_para, ori):
    neighbors = []
    [c_h, c_w] = ori_pos
    if ori == 0:
        for node in gen_nodes_para:
            if c_w - node[1] in range(-2, 3) and c_h > node[0]:
                neighbors.append(node)
    elif ori == 1:
        for node in gen_nodes_para:
            if (c_w - node[1] in range(-2, 3)) and (c_h < node[0]):
                neighbors.append(node)
    elif ori == 2:
        for node in gen_nodes_para:
            if c_h - node[0] in range(-2, 3) and c_w > node[1]:
                neighbors.append(node)
    elif ori == 3:
        for node in gen_nodes_para:
            if c_h - node[0] in range(-2, 3) and c_w < node[1]:
                neighbors.append(node)
    return neighbors

def get_avg_pos(node):
    pixel_num = len(node)
    node_array = np.array(node)

    [avg_h, avg_w] = np.sum(node_array, axis=0)
    avg_h = np.int8(round(avg_h / pixel_num))
    avg_w = np.int8(round(avg_w / pixel_num))
    return avg_h, avg_w

def get_output_nodes(output_copy):
    generated_nodes = []

    copy2 = np.zeros((120, 120), dtype=np.uint8)

    copy2[output_copy == 2] = 1
    for x in range(120):
        for y in range(120):
            if copy2[x, y] != 0:
                node = Extract_one_junction(copy2, x, y)
                generated_nodes.append(node)
    return generated_nodes

def Extract_one_junction(gen_copy, x, y):
    node = []
    node_array = []
    node_array.append([x, y])
    gen_copy[x, y] = 0

    while (len(node_array) != 0):
        point = node_array.pop(0)
        node.append(point)

        for pos_x in range(point[0] - 1, point[0] + 2):
            for pos_y in range(point[1] - 1, point[1] + 2):
                if pos_x in range(120) and pos_y in range(120):
                    if (gen_copy[pos_x, pos_y] != 0):
                        node_array.append([pos_x, pos_y])
                        gen_copy[pos_x, pos_y] = 0
    return node