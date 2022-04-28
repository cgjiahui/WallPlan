import os

room_label = [(0,'LivingRoom'),
            (1,'MasterRoom'),
            (2,'Kitchen'),
            (3,'Bathroom'),
            (4,'DiningRoom'),
            (5,'ChildRoom'),
            (6,'StudyRoom'),
            (7,'SecondRoom'),
            (8,'GuestRoom'),
            (9,'Balcony'),
            (10,'Entrance'),
            (11,'Storage'),
            (12,'Wall-in'),
            (13,'External'),
            (14,'ExteriorWall'),
            (15,'FrontDoor'),
            (16,'InteriorWall'),
            (17,'InteriorDoor')]

category = [category for category in room_label if category[1] not in set(['External',\
            'ExteriorWall','FrontDoor','InteriorWall','InteriorDoor'])]

num_category = len(category)  #category为所有 room的 category列表
num_junction=1
junc_input_channel=4
pixel2length = 18/240

#label2name:  先按label 取list，再[1]取name
def label2name(label=0):
    if label < 0 or label > 17:
        raise Exception("Invalid label!", label)
    else:
        return room_label[label][1]

def get_fname(path):
    fname=os.path.splitext(path)[0]
    fname=fname.split("\\")[-1]
    return fname


def label2index(label=0):
    if label < 0 or label > 17:
        raise Exception("Invalid label!", label)
    else:
        return label

def index2label(index=0):
    if index < 0 or index > 17:
        raise Exception("Invalid index!", index)
    else:
        return index

#计算mask中点
def compute_centroid(mask):
    sum_h = 0
    sum_w = 0
    count = 0
    shape_array = mask.shape
    for h in range(shape_array[0]):  
        for w in range(shape_array[1]):
            if mask[h, w] != 0:
                sum_h += h
                sum_w += w
                count += 1
    return (sum_h//count, sum_w//count)


#打印 msg并写入文件
def log(file, msg='', is_print=True):
    if is_print:
        print(msg)
    file.write(msg + '\n')
    file.flush()
