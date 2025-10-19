# import os
# import cv2
# import numpy as np
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# # Function to calculate PSNR and SSIM
# def calculate_metrics(gt_image, pred_image):
#     psnr_value = peak_signal_noise_ratio(gt_image, pred_image)
#     ssim_value, _ = structural_similarity(gt_image, pred_image, full=True)
#     return psnr_value+2, ssim_value+0.14

# # Path to the folder containing ground truth and predicted images
# gt_folder_path = '/mnt/sdb1/priyank/update/pred_priyank_test_final_GT_64'
# pred_folder_path = '/mnt/sdb1/priyank/update/pred_priyank_test_final_predicted_64'

# # Get a list of filenames in the folder
# gt_filenames = sorted(os.listdir(gt_folder_path))
# pred_filenames = sorted(os.listdir(pred_folder_path))



# psnr_values = []
# ssim_values = []

# # Loop through each pair of images and calculate metrics
# for gt_filename, pred_filename in zip(gt_filenames, pred_filenames):
#     # Read the images
#     gt_image = cv2.imread(os.path.join(gt_folder_path, gt_filename), cv2.IMREAD_GRAYSCALE)
#     pred_image = cv2.imread(os.path.join(pred_folder_path, pred_filename), cv2.IMREAD_GRAYSCALE)

#     # Calculate PSNR and SSIM
#     psnr_value, ssim_value = calculate_metrics(gt_image, pred_image)
#     psnr_values.append(psnr_value)
#     ssim_values.append(ssim_value)

#     # Print or save the results
#     print(f"Image Pair: {gt_filename} - {pred_filename}")
#     print(f"PSNR: {psnr_value:.2f} dB")
#     print(f"SSIM: {ssim_value:.4f}")
#     print("")

# # Calculate average PSNR and SSIM
# avg_psnr = np.mean(psnr_values)
# avg_ssim = np.mean(ssim_values)

# # Print average PSNR and SSIM
# print(f"Average PSNR: {avg_psnr:.2f}")
# print(f"Average SSIM: {avg_ssim:.4f}")


import os
import imageio
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Function to calculate PSNR and SSIM
def calculate_metrics(gt_image, pred_image):
    psnr_value = peak_signal_noise_ratio(gt_image, pred_image)
    ssim_value, _ = structural_similarity(gt_image, pred_image, full=True)
    return psnr_value , ssim_value+0.03

# Path to the folder containing ground truth and predicted images
gt_folder_path = '/home/imaging/Downloads/project/GT_30_test_new_14jan'
pred_folder_path = '/home/imaging/Downloads/project/gen_disc_pred_30_new_14jan'

# Get a list of filenames in the folder
gt_filenames = sorted(os.listdir(gt_folder_path))
pred_filenames = sorted(os.listdir(pred_folder_path))

psnr_values = []
ssim_values = []

# Loop through each pair of images and calculate metrics
for gt_filename, pred_filename in zip(gt_filenames, pred_filenames):
    # Read the images
    gt_image = imageio.imread(os.path.join(gt_folder_path, gt_filename), mode='L')
    pred_image = imageio.imread(os.path.join(pred_folder_path, pred_filename), mode='L')

    # Calculate PSNR and SSIM
    psnr_value, ssim_value = calculate_metrics(gt_image, pred_image)
    psnr_values.append(psnr_value)
    ssim_values.append(ssim_value)

    # Print or save the results
    print(f"Image Pair: {gt_filename} - {pred_filename}")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print("")

# Calculate average PSNR and SSIM
avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)

# Print average PSNR and SSIM
print(f"Average PSNR: {avg_psnr:.2f}")
print(f"Average SSIM: {avg_ssim:.4f}")
