###
### This file contains the main pipeline for training the GAN model.
### Functions are used from 'training.py', 'data_utils.py' and 'eval.py'.
### Last updated: 2022/05/04 9:15 AM
###


# Import external libraries
import os
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
# import pandas as pd

# Import from other files
from training import model_train
from data_utils import Dataset
from eval import mean_squared_error, peak_SNR, show_img, struct_simil_index
import random
import h5py
from sklearn.model_selection import train_test_split

# Set constants
# PATCH_SIZE       = (128,128)
MODEL_INPUT_SIZE = (512,512,1)
# IMAGE_SIZE       = (128,896)
# SAVE_PATH_MODELS = 'saved_models'
# SAVE_PATH_CSV    = os.path.join('logs','csv', datetime.now().strftime("%Y.%m.%d-%H.%M.%S") + '.csv')

# train = Dataset('data/RFdata_train.mat','RF_train_single', 'RF_train_avg', PATCH_SIZE, combine_n_frames=1, normalize=True, svd_denoise=['ref']) 
# val   = Dataset('data/RFdata_val.mat'  ,'RF_val_single'  , 'RF_val_avg'  , PATCH_SIZE, combine_n_frames=1, normalize=True, svd_denoise=['ref']) 



def split_train_test(idc, test_fraction):

    random.seed(0) # deterministic
    random.shuffle(idc) # in-place
    tmp = int((1-test_fraction)*len(idc))
    idc_train = idc[:tmp] # this makes a copy
    idc_test = idc[tmp:]
    return idc_train, idc_test

def normalize_data(X, axis = 0):
    mu = np.mean(X,axis =0)
    sigma = np.std(X, axis=0)

    X_norm = 2*(X-mu)/sigma
    return np.nan_to_num(X_norm)


# w_artifact_file = h5py.File('/mnt/sdb1/dataset/data/mice_sparse32_recon.mat', 'r')
w_artifact_file = h5py.File('/mnt/sdb1/priyank/GAN_PA_denoising/mice_sparse64_recon.mat', 'r')
variables_w = w_artifact_file.items()

for var in variables_w:
    name_w_artifact = var[0]
    data_w_artifact = var[1]
    if type(data_w_artifact) is h5py.Dataset:
        w_artifact = data_w_artifact
        # print(w_artifact.shape)
        w_artifact=normalize_data(w_artifact)
        # w_artifact = np.transpose(w_artifact, (1, 2, 0))





wo_artifact_file = h5py.File('/mnt/sdb1/priyank/GAN_PA_denoising/mice_full_recon.mat', 'r')

variables_wo = wo_artifact_file.items()

for var in variables_wo:
    name_wo_artifact = var[0]
    data_wo_artifact = var[1]
    if type(data_wo_artifact) is h5py.Dataset:
        wo_artifact = data_wo_artifact 
        # print(wo_artifact.shape)
        wo_artifact=normalize_data(wo_artifact)
        # wo_artifact = np.transpose(wo_artifact, (1, 2, 0))
       

print("hello", w_artifact.shape, wo_artifact.shape)



# # Set the random seed for reproducibility
# random_seed = 42

# # Split w_artifact
# w_artifact_train, w_artifact_test = train_test_split(w_artifact, test_size=0.2)

# # Split wo_artifact
# wo_artifact_train, wo_artifact_test = train_test_split(wo_artifact, test_size=0.2)

# # Print the shapes of the resulting datasets
# print("w_artifact_train shape:", w_artifact_train.shape)
# print("w_artifact_test shape:", w_artifact_test.shape)
# print("wo_artifact_train shape:", wo_artifact_train.shape)
# print("wo_artifact_test shape:", wo_artifact_test.shape)



# Train the model
# Gen, Disc, losses = model_train(train.patches, train.patches_ref, val.patches, val.patches_ref, epochs=500, img_shape=MODEL_INPUT_SIZE, batch_size=5)
Gen,Disc=model_train(w_artifact, wo_artifact, epochs=400, img_shape=MODEL_INPUT_SIZE, batch_size=1)
Gen.save('/mnt/sdb1/priyank/project/model_fit_64.h5')

# # Save the fully trained models
# Gen.save(os.path.join(SAVE_PATH_MODELS, f'Generator_{datetime.now().strftime("%Y%m%d-%H_%M_%S")}'))

# # Write losses to CSV-file
# pd.DataFrame(losses).to_csv(SAVE_PATH_CSV)





