import os
import cv2
import numpy as np
import random

path_src = "../images/Skipjack_Tuna/"
path_dest = "../images/Tuna_Species/Skipjack_tuna/"
dirs = os.listdir(path_src)
print(dirs)
randomlist = []
for i in range(0,len(dirs)):
	n = random.randrange(1,9,2)
	randomlist.append(n)
print(randomlist)
i=0
for items in dirs:
	item = items.split(".")[0]
	print(items)
	img = cv2.imread(path_src + items)
	img1 = cv2.resize(img, (224,224))
	img1 = cv2.medianBlur(img1,randomlist[i])
	cv2.imwrite(path_dest + item +"blur.png",img1)