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

def log(file, msg='', is_print=True):
    if is_print:
        print(msg)
    file.write(msg + '\n')
    file.flush()
