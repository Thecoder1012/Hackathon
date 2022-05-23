import numpy as np
import cv2
import os

def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output

#path_body = "../images/Actual_Pak_dataset/Silver/Silver/Body/"
#path_head = "../images/Actual_Pak_dataset/Silver/Silver/Head/"
#path_scales = "../images/Actual_Pak_dataset/Silver/Silver/Scales/"
path_src = "../images/Albacore_Tuna/"
path_dest = "../images/Albacore_Tuna_v2/"
#dirs = os.listdir(path_body) + os.listdir(path_head) + os.listdir(path_scales)
dirs = os.listdir(path_src)
print(dirs)
randomlist = []
for i in range(0,2 * len(dirs)):
	n = np.random.randint(1,224-170+1)
	randomlist.append(n)
print(randomlist)
i=0
for items in dirs:
	item = items.split(".")[0]
	img = cv2.imread(path_src + items)
	img1 = cv2.resize(img, (224,224))
	#img = cv2.imread(path_dest + item+".png")
	img1 = sp_noise(img1, 0.015)
	cv2.imwrite(path_dest+item+"noisy.png",img1)