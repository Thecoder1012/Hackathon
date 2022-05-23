import os
path_body = "../images/Actual_Pak_dataset/Catla/Catla/Body/"
path_head = "../images/Actual_Pak_dataset/Catla/Catla/Head/"
path_scales = "../images/Actual_Pak_dataset/Catla/Catla/Scales/"
dirs = os.listdir(path_body) + os.listdir(path_head) + os.listdir(path_scales)
print(dirs)