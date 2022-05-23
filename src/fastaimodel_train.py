import splitfolders
from fastai.vision import *
from fastai.metrics import error_rate, accuracy
import fastai
import cv2
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import torch
import torch.nn as nn
from fastai import *
from fastai.callbacks import *
import matplotlib.pyplot as plt

#split the data into train, val, test
splitfolders.ratio("./Tuna_Species_v2", output="Output", seed=1337, ratio=(.7,.2,.1), group_prefix=None)

print(fastai.__version__)
path=Path("./Output")
data = ImageDataBunch.from_folder(path, train='train', valid='val', ds_tfms=get_transforms(do_flip=False, p_affine=.20), size=224, bs=64, num_workers=8)
data.show_batch()

# See the classes and count of classes in your dataset
print(data.classes,data.c)
print(len(data.train_ds), len(data.valid_ds))
#learn = cnn_learner(data, models.mobilenet_v2, metrics = error_rate)
learn = cnn_learner(data, models.mobilenet_v2, metrics = [accuracy])
print(learn.model)
learn.fit_one_cycle(30, callbacks= SaveModelCallback(learn, every='epoch', monitor='accuracy'))
#model save
model_path=Path('./tunav2mobilenetv2e30acc').absolute()
learn.export(model_path / 'export.pkl')
learn.save(model_path / 'export')

#interplane plotting
interp = ClassificationInterpretation.from_learner(learn)

#see if it works
'''
losses,idxs = interp.top_losses()
len(dls.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix()
#upto here
'''
interp.plot_top_losses(9, figsize=(15,11))
#Confusion Matrix
#interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.plot_confusion_matrix()
plt.savefig('./tunav2mobilenetv2e30acc/confusion_matrix.png')
