#from google.colab import drive
#import warnings
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time
import pickle

from sklearn.neighbors import KNeighborsClassifier  
from sklearn import preprocessing
from tensorflow.keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
#from keras.layers.normalization import batch_normalization
from tensorflow.keras.layers import BatchNormalization
import seaborn as sns
from tensorflow.keras.applications.resnet50 import ResNet50
#drive.mount('/content/drive')
#warnings.filterwarnings("ignore", category=FutureWarning)
#!pip3 install split-folders tqdm
import splitfolders  # or import split_folders
#splitfolders.ratio("../images/Tuna_Species_v2", output="Output_Main", seed=1337, ratio=(.7,.2,.1), group_prefix=None) # default values
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

train_images = []
train_labels = []
for path in glob.glob("Output_Main/train/*"):
  label = path.split("\\")[-1]
  print(label)
  for img_path in glob.glob(os.path.join(path, "*.png")):
    #print(img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    train_images.append(img)
    train_labels.append(label)
train_images = np.array(train_images)
train_labels = np.array(train_labels)

val_images = []
val_labels = []
for path in glob.glob("Output_Main/val/*"):
  label = path.split("\\")[-1]
  print(label)
  for img_path in glob.glob(os.path.join(path, "*.png")):
    #print(img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    val_images.append(img)
    val_labels.append(label)
val_images = np.array(val_images)
val_labels = np.array(val_labels)

test_images = []
test_labels = []
for path in glob.glob("Output_Main/test/*"):
  label = path.split("\\")[-1]
  print(label)
  for img_path in glob.glob(os.path.join(path, "*.png")):
    #print(img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    test_images.append(img)
    test_labels.append(label)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(val_labels)
val_labels_encoded = le.transform(val_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)


x_train, y_train, x_val, y_val, x_test, y_test = train_images, train_labels_encoded, val_images, val_labels_encoded, test_images, test_labels_encoded

x_train, x_val, x_test = x_train / 255.0, x_val / 255.0, x_test / 255.0
print(x_train.shape)
resnet_model =  ResNet50(weights = "imagenet", include_top = False, input_shape = (224,224,3))
for layer in resnet_model.layers:
  layer.trainable = False
resnet_model.summary()

#Feature Extractor part Resnet50 network

feature_extractor = resnet_model.predict(x_train)
features = feature_extractor.reshape(feature_extractor.shape[0], -1)
x_for_training  = features

#ML Model
model_rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
model_xg = XGBClassifier()
model_dt = DecisionTreeClassifier(random_state=0)
model_svm = svm.SVC()
model_knn= KNeighborsClassifier(n_neighbors=100, metric='minkowski', p=2 )  

'''
model_rf.fit(x_for_training, y_train)
print("random forest train fit DONE!")

#Testing on Val
x_val_feature = resnet_model.predict(x_val)
x_val_features = x_val_feature.reshape(x_val_feature.shape[0], -1)
prediction = model_rf.predict(x_val_features)
prediction = le.inverse_transform(prediction)
print("Accuracy:", metrics.accuracy_score(val_labels, prediction))
cm = confusion_matrix(val_labels, prediction)
cv2.imwrite("val_rf.png", cm)
sns.heatmap(cm, annot = True)
print("random forest val DONE!")
#Testing on Test Set
x_test_feature = resnet_model.predict(x_test)
x_test_features = x_test_feature.reshape(x_test_feature.shape[0], -1)
prediction = model_rf.predict(x_test_features)
prediction = le.inverse_transform(prediction)
print("Test Accuracy:", metrics.accuracy_score(test_labels, prediction))
cm = confusion_matrix(test_labels, prediction)
cv2.imwrite("test_rf.png", cm)
sns.heatmap(cm, annot = True)
print("random forest test DONE!")

print("Saving RF model")
filename = 'model_rf.sav'
pickle.dump(model_rf, open(filename, 'wb'))

'''
model_xg.fit(x_for_training, y_train)
print("Xgboost train fit DONE!")

#Testing on Val
x_val_feature = resnet_model.predict(x_val)
x_val_features = x_val_feature.reshape(x_val_feature.shape[0], -1)
prediction = model_xg.predict(x_val_features)
prediction = le.inverse_transform(prediction)
print("Accuracy:", metrics.accuracy_score(val_labels, prediction))
cm = confusion_matrix(val_labels, prediction)
cv2.imwrite("val_xg.png", cm)
sns.heatmap(cm, annot = True)
print("xgboost val DONE!")

#Testing on Test Set
x_test_feature = resnet_model.predict(x_test)
x_test_features = x_test_feature.reshape(x_test_feature.shape[0], -1)
prediction = model_xg.predict(x_test_features)
prediction = le.inverse_transform(prediction)
print("Test Accuracy:", metrics.accuracy_score(test_labels, prediction))
cm = confusion_matrix(test_labels, prediction)
cv2.imwrite("test_xg.png", cm)
sns.heatmap(cm, annot = True)
print("xgboost test DONE!")

print("Saving XG model")
filename = 'model_xg.sav'
pickle.dump(model_xg, open(filename, 'wb'))
'''

model_dt.fit(x_for_training, y_train)
print("decision tree train fit DONE!")

#Testing on Val
x_val_feature = resnet_model.predict(x_val)
x_val_features = x_val_feature.reshape(x_val_feature.shape[0], -1)
prediction = model_dt.predict(x_val_features)
prediction = le.inverse_transform(prediction)
print("Accuracy:", metrics.accuracy_score(val_labels, prediction))
cm = confusion_matrix(val_labels, prediction)
cv2.imwrite("val_dt.png", cm)
sns.heatmap(cm, annot = True)
print("decision tree val DONE!")

#Testing on Test Set
x_test_feature = resnet_model.predict(x_test)
x_test_features = x_test_feature.reshape(x_test_feature.shape[0], -1)
prediction = model_dt.predict(x_test_features)
prediction = le.inverse_transform(prediction)
print("Test Accuracy:", metrics.accuracy_score(test_labels, prediction))
cm = confusion_matrix(test_labels, prediction)
cv2.imwrite("test_dt.png", cm)
sns.heatmap(cm, annot = True)
print("decision tree test DONE!")

print("Saving DT model")
filename = 'model_dt.sav'
pickle.dump(model_dt, open(filename, 'wb'))

model_svm.fit(x_for_training, y_train)
print("Svm train fit DONE!")

#Testing on Val
x_val_feature = resnet_model.predict(x_val)
x_val_features = x_val_feature.reshape(x_val_feature.shape[0], -1)
prediction = model_svm.predict(x_val_features)
prediction = le.inverse_transform(prediction)
print("Accuracy:", metrics.accuracy_score(val_labels, prediction))
cm = confusion_matrix(val_labels, prediction)
cv2.imwrite("val_svm.png", cm)
sns.heatmap(cm, annot = True)
print("Svm val DONE!")

#Testing on Test Set
x_test_feature = resnet_model.predict(x_test)
x_test_features = x_test_feature.reshape(x_test_feature.shape[0], -1)
prediction = model_svm.predict(x_test_features)
prediction = le.inverse_transform(prediction)
print("Test Accuracy:", metrics.accuracy_score(test_labels, prediction))
cm = confusion_matrix(test_labels, prediction)
cv2.imwrite("test_svm.png", cm)
sns.heatmap(cm, annot = True)
print("Svm test DONE!")

print("Saving SVM model")
filename = 'model_svm.sav'
pickle.dump(model_svm, open(filename, 'wb'))
'''

model_knn.fit(x_for_training, y_train)
print("knn train fit DONE!")

#Testing on Val
x_val_feature = resnet_model.predict(x_val)
x_val_features = x_val_feature.reshape(x_val_feature.shape[0], -1)
prediction = model_knn.predict(x_val_features)
prediction = le.inverse_transform(prediction)
print("Accuracy:", metrics.accuracy_score(val_labels, prediction))
cm = confusion_matrix(val_labels, prediction)
cv2.imwrite("val_knn.png", cm)
sns.heatmap(cm, annot = True)
print("knn val DONE!")

#Testing on Test Set
x_test_feature = resnet_model.predict(x_test)
x_test_features = x_test_feature.reshape(x_test_feature.shape[0], -1)
prediction = model_knn.predict(x_test_features)
prediction = le.inverse_transform(prediction)
print("Test Accuracy:", metrics.accuracy_score(test_labels, prediction))
cm = confusion_matrix(test_labels, prediction)
cv2.imwrite("test_knn.png", cm)
sns.heatmap(cm, annot = True)
print("knn test DONE!")

print("Saving SVM model")
filename = 'model_knn.sav'
pickle.dump(model_knn, open(filename, 'wb'))
