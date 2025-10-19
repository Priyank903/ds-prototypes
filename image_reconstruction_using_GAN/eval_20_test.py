import os
import numpy as np
import imageio
import tensorflow as tf
from matplotlib import pyplot as plt
import h5py
from tqdm import tqdm
from augmentation_main import Gen as gen
from augmentation_main import wo_artifact_test,w_artifact_test


# # from datetime import datetime
# # from time import time
# # from keras import backend as K_backend
# # K_backend.set_image_data_format('channels_last')
# # tf.config.run_functions_eagerly(True)

# # model = tf.keras.models.load_model('/mnt/sdb1/priyank/project/model_fit_64.h5')
# # # model.compile(optimizer='adam',  
# # #               loss='categorical_crossentropy',  
# # #               metrics=['accuracy']) 

# # folder_path = "/mnt/sdb1/priyank/project/arti_20_test"

# # i=0


# # for filename in os.listdir(folder_path):
# #     i+=1
# #     img_path = os.path.join(folder_path, filename)
# #     img = imageio.imread(img_path)
# #     print(img.shape)
#     # img = img.asty/pe('float32')
#     # img = tf.reshape(img, (1, 512, 512, 1))
#     # gen_output = model.predict(img)
#     # plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
#     # plt.imshow(gen_output[0],cmap='gray')  # Use an appropriate colormap
#     # plt.title(f'gen_pred_test_20')
#     # plt.savefig(f'/mnt/sdb1/priyank/project/gen_pred_20_test/test_pred_20_{i}.png', format='png', bbox_inches='tight')
#     # plt.show()
# def normalize_data(X, axis = 0):
#     mu = np.mean(X,axis =0)
#     sigma = np.std(X, axis=0)

#     X_norm = 2*(X-mu)/sigma
#     return np.nan_to_num(X_norm)


# # w_artifact_file = h5py.File('/mnt/sdb1/dataset/data/mice_sparse32_recon.mat', 'r')
# w_artifact_file = h5py.File('//mnt/sdb1/priyank/project/mice_sparse64_recon.mat', 'r')
# variables_w = w_artifact_file.items()

# for var in variables_w:
#     name_w_artifact = var[0]
#     data_w_artifact = var[1]
#     if type(data_w_artifact) is h5py.Dataset:
#         w_artifact = data_w_artifact
#         # print(w_artifact.shape)
#         w_artifact=normalize_data(w_artifact)
#         # w_artifact = np.transpose(w_artifact, (1, 2, 0))





# wo_artifact_file = h5py.File('/mnt/sdb1/priyank/project/mice_full_recon.mat', 'r')

# variables_wo = wo_artifact_file.items()

# for var in variables_wo:
#     name_wo_artifact = var[0]
#     data_wo_artifact = var[1]
#     if type(data_wo_artifact) is h5py.Dataset:
#         wo_artifact = data_wo_artifact 
#         # print(wo_artifact.shape)
#         wo_artifact=normalize_data(wo_artifact)
#         # wo_artifact = np.transpose(wo_artifact, (1, 2, 0))
       

# print("hello", w_artifact.shape, wo_artifact.shape)

def batch_generator(x, n):
    """
    X: data
    n: batch size
    """
    start, stop = 0, n
    while True:
        if start < stop:
            yield x[start:stop]
        else:
            break
        start = stop
        stop = (stop + n) % len(x)


batches_X_test = batch_generator(w_artifact_test, 1)
batches_y_test = batch_generator(wo_artifact_test, 1)
iteration=0

for image_batch, ref_batch in tqdm(zip(batches_X_test, batches_y_test)):

    # Change data types for compatibility
    image_batch = image_batch.astype('float32')
    # ref_batch   = ref_batch.astype('float32')
    image_batch = tf.reshape(image_batch, (1, 512, 512, 1))
    # ref_batch = tf.reshape(ref_batch,(1, 512, 512, 1))
    out=gen.predict(image_batch)
    plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
    plt.imshow(out[0],cmap='gray')  # Use an appropriate colormap
    plt.title(f'gen_pred_20')
    plt.savefig(f'/mnt/sdb1/priyank/project/gen_disc_pred_20_new_12jan/pred_{iteration}.png', format='png', bbox_inches='tight')
    plt.show()
    iteration+=1

        

        


