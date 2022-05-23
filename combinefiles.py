import os
from PIL import Image
import random

def get_concat_h_multi_resize(im_list, resample=Image.BICUBIC):
    min_height = min(im.height for im in im_list)
    im_list_resize = [im.resize((int(im.width * min_height / im.height), min_height),resample=resample)
                      for im in im_list]
    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new('RGB', (total_width, min_height))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    return dst

def get_concat_v_multi_resize(im_list, resample=Image.BICUBIC):
    min_width = min(im.width for im in im_list)
    im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)),resample=resample)
                      for im in im_list]
    total_height = sum(im.height for im in im_list_resize)
    dst = Image.new('RGB', (min_width, total_height))
    pos_y = 0
    for im in im_list_resize:
        dst.paste(im, (0, pos_y))
        pos_y += im.height
    return dst

def get_concat_tile_resize(im_list_2d, resample=Image.BICUBIC):
    im_list_v = [get_concat_h_multi_resize(im_list_h, resample=resample) for im_list_h in im_list_2d]
    return get_concat_v_multi_resize(im_list_v, resample=resample)

for i in range(8000,9000):
	filenames = random.sample(os.listdir(orig_sourceTR), 8)
	im1 = Image.open(orig_sourceTR + filenames[0])
	im2 = Image.open(orig_sourceTR + filenames[1])
	im3 = Image.open(orig_sourceTR + filenames[2])
	im4 = Image.open(orig_sourceTR + filenames[3])
	im5 = Image.open(orig_sourceTR + filenames[4])
	im6 = Image.open(orig_sourceTR + filenames[5])
	im7 = Image.open(orig_sourceTR + filenames[6])
	im8 = Image.open(orig_sourceTR + filenames[7])
	#im9 = Image.open(orig_sourceTR + filenames[8])

	image_list = [[im1,im2,im3,im4], [im5, im6, im7, im8]]
	#random.shuffle(image_list)
	get_concat_tile_resize(image_list).save('../images/merged_data/Original/'+str(i)+'.png')
	
	#mask concat
	im1 = Image.open(mask_sourceTR + filenames[0])
	im2 = Image.open(mask_sourceTR + filenames[1])
	im3 = Image.open(mask_sourceTR + filenames[2])
	im4 = Image.open(mask_sourceTR + filenames[3])
	im5 = Image.open(mask_sourceTR + filenames[4])
	im6 = Image.open(mask_sourceTR + filenames[5])
	im7 = Image.open(mask_sourceTR + filenames[6])
	im8 = Image.open(mask_sourceTR + filenames[7])
	#im9 = Image.open(mask_sourceTR + filenames[8])

	image_list = [[im1,im2,im3,im4], [im5, im6, im7, im8]]
	#random.shuffle(image_list)
	get_concat_tile_resize(image_list).save('../images/merged_data/Mask/'+str(i)+'.png')