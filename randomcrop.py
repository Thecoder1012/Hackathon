import os
import cv2
import numpy as np
import random

def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    #assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]

#path_body = "../images/Actual_Pak_dataset/Cyprinus carpio/Cyprinus carpio/Body/"
#path_head = "../images/Actual_Pak_dataset/Cyprinus carpio/Cyprinus carpio/Head/"
path_src = "../images/Albacore_Tuna/"
path_dest = "../images/Albacore_Tuna_v2/"
#dirs = os.listdir(path_body) + os.listdir(path_head) + os.listdir(path_scales)
dirs = os.listdir(path_src)
print(dirs)
randomlist = []
for i in range(0,2 * len(dirs)):
	n = random.randint(1,224-170+1)
	randomlist.append(n)
print(randomlist)
i=0
for items in dirs:
	item = items.split(".")[0]
	#img = cv2.imread(path_dest + item +".png")
	print(items)
	img = cv2.imread(path_src + items)
	#cv2.imshow("image",img)
	#cv2.waitKey(0)
	img1 = cv2.resize(img, (224,224))
	height, width = img1.shape[0], img1.shape[1]
	dy, dx = (170,170)
	x = randomlist[i]
	y = randomlist[i+1]
	i = i+1
	random_crop_img = img1[y:(y+dy), x:(x+dx), :]
	cv2.imwrite(path_dest + item +"crop.png",random_crop_img)
cv2.waitKey(0)