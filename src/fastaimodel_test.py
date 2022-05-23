#FOR DETECTION ############################

import fastai; fastai.__version__
from fastai.vision import *
from fastai.metrics import error_rate, accuracy
from fastai.callbacks.hooks import *
import os
import cv2
import numpy as np
import warnings
import time

warnings.filterwarnings("ignore", category=FutureWarning)

pathsrc = './download.png' #change your source picture path here

pathdst = "./Try/" #change your destination path here

def model():
#Loading the Trained Model
	learn=load_learner("./tunav2mobilenetv2e30acc")
	model=learn.model.eval()
	path = "./Output/test/Albacore_Tuna/"
	dirs = os.listdir(path)
	list1=[]
	for item in dirs:
  		if os.path.isfile(path+item):
		    img=open_image(path+item)
		    print(item)
		    img.resize(224)
		    pred=learn.predict(img)
		    cls=int(pred[1])
		    #print(pred,cls)
		    #print("Result: ")
		    '''
		    if(cls==0):
		      print("Catla")
		    elif(cls==1):
		      print("Cyprinus Carpio")
		    elif(cls==2):
		      print("Grass Carp")
		    elif(cls==3):
		      print("Mori")
		    elif(cls==4):
		      print("Rohu")
		    else:
		      print('Silver')
		    if(cls==0):
		      print("Albacore")
		    elif(cls==1):
		      print("Big Eye")
		    elif(cls==2):
		      print("Frigate")
		    elif(cls==3):
		      print("Kawakawa")
		    elif(cls==4):
		      print("Skipjack")
		    else:
		      print('Yellowfin')
		    '''
		    #print(max(pred[2]*100))
		    if (cls != 0):
		    	if(cls==0):
		      		print("Albacore")
		    	elif(cls==1):
		      		print("Big Eye")
		    	elif(cls==2):
		      		print("Frigate")
		    	elif(cls==3):
		      		print("Kawakawa")
		    	elif(cls==4):
		      		print("Skipjack")
		    	else:
		      		print('Yellowfin')
		    	print(max(pred[2]*100))
		    	time.sleep(5)
		    else :
		    	list1.append(max(pred[2]*100))
		    #img
	avgscore = sum(list1) / len(dirs)
	print("Average score:", avgscore)    
    # Read image
if __name__ == '__main__' :
	model()

