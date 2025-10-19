import sklearn
from sklearn import metrics
import sys
import glob
import re
import os
import time
import numpy as np
import random
import h5py
import matplotlib.pyplot as plt
from numpy import load
import tensorflow as tf
import cv2 
# from scipy.ndimage import rotate 

import imageio
from skimage.transform import resize, rotate
from skimage import img_as_ubyte  # For converting floating-point images to uint8

import subprocess
import time



# import pandas as pd

# Import from other files
from training import model_train


def normalize_data(X, axis = 0):
    mu = np.mean(X,axis =0)
    sigma = np.std(X, axis=0)

    X_norm = 2*(X-mu)/sigma
    return np.nan_to_num(X_norm)


def augmentation(image, imageB, org_width=160,org_height=224, width=190, height=262):
    max_angle=5
    image=cv2.resize(image,(height,width))
    imageB=cv2.resize(imageB,(height,width))

    angle=np.random.randint(max_angle)
    if np.random.randint(2):
        angle=-angle
    
    image=rotate(image,angle)
    imageB=rotate(imageB,angle)

    # image=image.rotate(angle,resize=True)
    # imageB=imageB.rotate(angle,resize=True)

    xstart=np.random.randint(width-org_width)
    ystart=np.random.randint(height-org_height)
    image=image[xstart:xstart+org_width,ystart:ystart+org_height]
    imageB=imageB[xstart:xstart+org_width,ystart:ystart+org_height]

    if np.random.randint(2):
        image=cv2.flip(image,1)
        imageB=cv2.flip(imageB,1)

    if np.random.randint(2):
        image=cv2.flip(image,0)
        imageB=cv2.flip(imageB,0)

    image=cv2.resize(image,(512,512))
    imageB=cv2.resize(imageB,(512,512))

    return image,imageB




def split_train_test(idc, test_fraction):

    random.seed(0) # deterministic
    random.shuffle(idc) # in-place
    tmp = int((1-test_fraction)*len(idc))
    idc_train = idc[:tmp] # this makes a copy
    idc_test = idc[tmp:]
    return idc_train, idc_test




# w_artifact_file = h5py.File('/mnt/sdb1/dataset/data/mice_sparse32_recon.mat', 'r')
w_artifact_file = h5py.File('/home/imaging/Downloads/project/mice_sparse64_recon.mat', 'r')
variables_w = w_artifact_file.items()
for var in variables_w:
    name_w_artifact = var[0]
    data_w_artifact = var[1]
    print(data_w_artifact.shape, name_w_artifact)
    if type(data_w_artifact) is h5py.Dataset:
        w_artifact = data_w_artifact
        w_artifact=normalize_data(w_artifact)




wo_artifact_file = h5py.File('/home/imaging/Downloads/project/mice_full_recon.mat', 'r')

variables_wo = wo_artifact_file.items()

for var in variables_wo:
    name_wo_artifact = var[0]
    data_wo_artifact = var[1]
    if type(data_wo_artifact) is h5py.Dataset:
        wo_artifact = data_wo_artifact # NumPy ndArray / Value
        wo_artifact=normalize_data(wo_artifact)


for i in range(274):
    i1, i2 = augmentation(w_artifact[i], wo_artifact[i])
    i1 = np.reshape(i1, (1, 512, 512))
    i2 = np.reshape(i2, (1, 512, 512))

    


    w_artifact = np.concatenate((w_artifact, i1), axis=0)
    wo_artifact = np.concatenate((wo_artifact, i2), axis=0)




print("hello", w_artifact.shape, wo_artifact.shape)



# Set the random seed for reproducibility
random_seed = 42

# Split w_artifact
w_artifact_train, w_artifact_test = split_train_test(w_artifact, 0.3)

# Split wo_artifact
wo_artifact_train, wo_artifact_test = split_train_test(wo_artifact, 0.3)

# Print the shapes of the resulting datasets
print("w_artifact_train shape:", w_artifact_train.shape)
print("w_artifact_test shape:", w_artifact_test.shape)
print("wo_artifact_train shape:", wo_artifact_train.shape)
print("wo_artifact_test shape:", wo_artifact_test.shape)
        



# for i in range(438):
#     plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
#     plt.imshow(wo_artifact_train[i],cmap='gray')  # Use an appropriate colormap
#     plt.title(f'GT')
#     plt.savefig(f'/mnt/sdb1/priyank/project/GT_80_train_new/GT_{i}.png', format='png', bbox_inches='tight')
#     plt.show()

#     plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
#     plt.imshow(w_artifact_train[i],cmap='gray')  # Use an appropriate colormap
#     plt.title(f'aug')
#     plt.savefig(f'/mnt/sdb1/priyank/project/arti_80_train_new/arti_{i}.png', format='png', bbox_inches='tight')
#     plt.show()

# for i in range(110):
#     plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
#     plt.imshow(wo_artifact_test[i],cmap='gray')  # Use an appropriate colormap
#     plt.title(f'GT')
#     plt.savefig(f'/mnt/sdb1/priyank/project/GT_20_test_new/GT_{i}.png', format='png', bbox_inches='tight')
#     plt.show()

#     plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
#     plt.imshow(w_artifact_test[i],cmap='gray')  # Use an appropriate colormap
#     plt.title(f'aug')
#     plt.savefig(f'/mnt/sdb1/priyank/project/arti_20_test_new/arti_{i}.png', format='png', bbox_inches='tight')
#     plt.show()

MODEL_INPUT_SIZE = (512,512,1)

Gen,Disc=model_train(w_artifact_train, wo_artifact_train, w_artifact_test,wo_artifact_test,epochs=100, img_shape=MODEL_INPUT_SIZE, batch_size=1)
Gen.save('/mnt/sdb1/priyank/project/gen_fit_64_70_14jan.h5')
Disc.save('/mnt/sdb1/priyank/project/disc_fit_64_70_14jan.h5')

