# Le Minh Bang
# 313540015
# EECS IGP

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


# To visualize on image likes .PNG, .JPEG
def process_data(channel):
    channel = np.clip(channel, 0, 255)  # Make values in range from 0 to 255
    return channel.astype(np.uint8)  # Change value type to uint8


# Function define grayscale image
def get_grayscale_image(image_name):
    image = cv2.imread(image_name)
    image = cv2.resize(image, (256, 256))
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    return Y


# Calculate 2D-DCT II Formula
def get_2d_dct(block):
    N = block.shape[0]
    F = np.zeros_like(block)

    for u in tqdm(range(N), desc="dct2d_process"):
        for v in range(N):
            C_u = 1 / np.sqrt(2) if u == 0 else 1
            C_v = 1 / np.sqrt(2) if v == 0 else 1

            sum = 0.0
            for x in range(N):
                for y in range(N):
                    sum += (
                        block[x, y]
                        * np.cos((2 * x + 1) * np.pi * u / (2 * N))
                        * np.cos((2 * y + 1) * np.pi * v / (2 * N))
                    )

            F[u, v] = sum * C_u * C_v * (2 / N)
    return F


# Calulate 2D-IDCT
def get_2d_idct(coe_block):
    N = coe_block.shape[0]
    F_reconstructed = np.zeros_like(coe_block)

    for x in tqdm(range(N), desc="idct2d_process"):
        for y in range(N):
            sum = 0.0
            for u in range(N):
                for v in range(N):
                    C_u = 1 / np.sqrt(2) if u == 0 else 1
                    C_v = 1 / np.sqrt(2) if v == 0 else 1

                    sum += (
                        C_u
                        * C_v
                        * coe_block[u, v]
                        * np.cos((2 * x + 1) * u * np.pi / (2 * N))
                        * np.cos((2 * y + 1) * v * np.pi / (2 * N))
                    )

            F_reconstructed[x, y] = sum * (2 / N)
    return F_reconstructed


# Calculate 1D-DCT
def get_1d_dct(list_1d):
    coe_1d = np.zeros(list_1d.shape)
    N = list_1d.shape[0]
    for u in range(N):
        sum = 0.0
        C_u = 1 / np.sqrt(2) if u == 0 else 1
        for x in range(N):
            sum += list_1d[x] * np.cos((2 * x + 1) * u * np.pi / (2 * N))
        coe_1d[u] = sum * C_u * np.sqrt(2 / N)

    return coe_1d


# Calculate 1D-IDCT
def get_1d_idct(coe_1d):
    list_1d = np.zeros_like(coe_1d)
    N = coe_1d.shape[0]
    for x in range(N):
        sum = 0.0
        for u in range(N):
            C_u = 1 / np.sqrt(2) if u == 0 else 1
            sum += C_u * coe_1d[u] * np.cos((2 * x + 1) * u * np.pi / (2 * N))
        list_1d[x] = sum * np.sqrt(2 / N)

    return list_1d


# Calculate PSNR
def get_psnr(expect, actual):
    mse = np.mean((expect - actual) ** 2)
    if mse == 0:
        return 100
    # Grayscale image with 8 bit
    print(f"MSE = {mse}")
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel * max_pixel) / mse)
    return psnr

# Convert to log domain
def convert_to_log_domain(coe_values):
    return np.log1p(np.abs(coe_values))

# Helper function
def save_result(data, saved_file_name):
    np.save(f"./results/{saved_file_name}", data)

def save_time_log(duration, time_log_name):
    fp = open(f"./results/{time_log_name}.txt", "w")
    fp.write(f"Processing time is: {duration}")
    fp.close()

def save_psnr(value, psnr_log_name):
    fp = open(f"./results/{psnr_log_name}.txt", "w")
    fp.write(f"PSNR is: {value}")
    fp.close()

def extract_image(save_file_name, load_file_name):
    cv2.imwrite(
        f"./images/{save_file_name}.png", np.load(f"./results/{load_file_name}.npy")
    )


# BLOCK 1: Get image and Get number of rows or columns
image = get_grayscale_image("lena.png")
N = image.shape[0]

# # BLOCK 2: Get 1D DCT coe, 1D IDCT, and PSNR - Comment or uncomment this to use
# """ 
#     Input: N, image (image is resized for better performance because of my pc)
#     Output: 1D_DCT_COE.npy (An array of coefficient values), 1D_IDCT.npy (grayscale image), PSNR value
# """
# # 1D-DCT --------------------------------------------------
# start_at = time.process_time()

# d1_dct_coe = np.zeros_like(image)
# for i in tqdm(range(N), desc="dct1d_x"):
#     d1_dct_coe[i, :] = get_1d_dct(image[i, :])
# for i in tqdm(range(N), desc="dct1d_y"):
#     d1_dct_coe[:, i] = get_1d_dct(d1_dct_coe[:, i])

# save_result(d1_dct_coe, "1D_DCT_COE")

# end_at = time.process_time()
# duration = end_at - start_at
# print(f"\nProcess time is: {duration}")
# save_time_log(duration, "1D_DCT_COE_TIME_LOG")

# # 1D-IDCT --------------------------------------------------
# start_at = time.process_time()

# d1_dct_coe = np.load("./results/1D_DCT_COE.npy")
# d1_idct = np.zeros_like(d1_dct_coe)

# for i in tqdm(range(N), desc="idct1d_x"):
#     d1_idct[i, :] = get_1d_idct(d1_dct_coe[i, :])
# for i in tqdm(range(N), desc="idct1d_y"):
#     d1_idct[:, i] = get_1d_idct(d1_idct[:, i])

# save_result(d1_idct, "1D_IDCT")
# end_at = time.process_time()
# duration = end_at - start_at
# print(f"\nProcess time is: {duration}")
# save_time_log(duration, "1D_IDCT_TIME_LOG")

# # PSNR -----------------------------------------------------
# psnr = get_psnr(image, np.load('./results/1D_IDCT.npy').astype(int))
# save_psnr(psnr, "1D_DCT_PSNR")

# # Visualize ------------------------------------------------
# result_images = [
#     {"result": image, "descr": "Original Image"},
#     {"result": convert_to_log_domain(np.load("./results/1D_DCT_COE.npy")), "descr": "Coe Values (Log domain)"},
#     {"result": np.load("./results/1D_IDCT.npy"), "descr": "Reconstructed Image"},
# ]

# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# for i in range(3):
#     result_image = result_images[i]
#     axes[i].imshow(result_image["result"], cmap="gray")
#     axes[i].set_title(result_image["descr"])

# plt.show()


# # BLOCK 3: 2D-DCT, 2D-IDCT, PSNR
# """ 
#     Input: N, image (image is resized for better performance because of my pc)
#     Output: 2D_DCT_COE.npy (An array of coefficient values), 2D_IDCT.npy (grayscale image), PSNR value
# """
# # 2D-DCT --------------------------------------------------
# start_at = time.process_time()

# d2_dct_coe = get_2d_dct(image)

# save_result(d2_dct_coe, "2D_DCT_COE")

# end_at = time.process_time()
# duration = end_at - start_at
# print(f"\nProcess time is: {duration}")
# save_time_log(duration, "2D_DCT_COE_TIME_LOG")

# # 2D-IDCT --------------------------------------------------
# start_at = time.process_time()

# d2_dct_coe = np.load("./results/2D_DCT_COE.npy")

# d2_idct = get_2d_idct(d2_dct_coe)

# save_result(d2_idct, "2D_IDCT")
# end_at = time.process_time()
# duration = end_at - start_at
# print(f"\nProcess time is: {duration}")
# save_time_log(duration, "2D_IDCT_TIME_LOG")

# # PSNR -----------------------------------------------------
# psnr = get_psnr(image, np.load('./results/2D_IDCT.npy').astype(int))
# save_psnr(psnr, "2D_DCT_PSNR")

# # Visualize ------------------------------------------------
# result_images = [
#     {"result": image, "descr": "Original Image"},
#     {"result": convert_to_log_domain(np.load("./results/2D_DCT_COE.npy")), "descr": "Coe Values (Log domain)"},
#     {"result": np.load("./results/2D_IDCT.npy"), "descr": "Reconstructed Image"},
# ]

# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# for i in range(3):
#     result_image = result_images[i]
#     axes[i].imshow(result_image["result"], cmap="gray")
#     axes[i].set_title(result_image["descr"])

# plt.show()