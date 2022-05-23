# importing required packages
from pathlib import Path
import shutil
import os

# defining source and destination
# paths
# path to source directory
src_dir = '../images/Segmentation_Dataset\Fish_Dataset\Fish_Dataset/Shrimp/Shrimp/'

# path to destination directory
dest_dir = '../images/HaarCascade_Train/Positive/'

files=os.listdir(src_dir)

# iterating over all the files in
# the source directory
for fname in files:
	
	# copying the files to the
	# destination directory
	shutil.copy2(os.path.join(src_dir,fname), dest_dir)
