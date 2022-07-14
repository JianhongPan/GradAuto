import os
import numpy as np
import cv2

from utils import ListAverageMeter, AverageMeter, more_config

mean = np.asarray([[[[0.4914]], [[0.4822]], [[0.4465]]]])
std = np.asarray([[[[0.2023]], [[0.1994]], [[0.2010]]]])

def edge_count(img_path):
    img = np.load(img_path)
    img = (img * std + mean) * 255
    img = img.squeeze()
    img = img.transpose(1,2,0)
    img = img.astype(np.uint8)
    edge = cv2.Canny(img,300,600)
    edge = (edge == 255)
    return edge.sum()

type = "sp"
k = "10.0"

path = "./numpy_output/ours_"+type+"_k_"+k
# path = "./numpy_output/ilfo_"+type
paths = os.listdir(path)

edge_nums = AverageMeter()

for idx, img_path in enumerate(paths):
    if img_path.find("mod"):
        edge_num = edge_count(path+'/'+img_path)
        edge_num_ori = edge_count(path+'/'+img_path.replace("ori","mod"))
        edge_nums.update(abs(edge_num_ori-edge_num))
        if idx%10 == 0:
            print(edge_nums.avg)

    
