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
    channel = np.clip(channel, 0, 255) # Make values in range from 0 to 255 
    return channel.astype(np.uint8) # Change value type to uint8

# Function define grayscale image
def get_grayscale_image(image_name):
    image = cv2.imread(image_name)
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    return Y

# Calculate 2D-DCT II Formula
def get_2d_dct(block):
    N = block.shape[0]
    F = np.zeros_like(block)

    for u in range(N):
        for v in range(N):
            C_u = 1/np.sqrt(2) if u == 0 else 1
            C_v = 1/np.sqrt(2) if v == 0 else 1

            sum = 0.0
            for x in range(N):
                for y in range(N):
                    sum += block[x,y] + np.cos((2*x + 1)*np.pi*u / (2*N))*np.cos((2*y + 1)*np.pi*v / (2*N))

            F[u,v] = sum * C_u * C_v * (2/N)
    return F

# Calulate 2D-IDCT
def get_2d_idct(coe_block):
    N = coe_block.shape[0] 
    F_reconstructed = np.zeros_like(coe_block)

    for x in range(N):
        for y in range(N):
            sum = 0.0
            for u in range(N):
                for v in range(N):
                    C_u = 1/np.sqrt(2) if u == 0 else 1
                    C_v = 1/np.sqrt(2) if v == 0 else 1

                    sum += C_u + C_v * coe_block[u,v]*np.cos((2*x+1)*u*np.pi / (2*N))*np.cos((2*y+1)*v*np.pi / (2*N))

            F_reconstructed[x,y] = sum * (2/N)
    return F_reconstructed

# Calculate 1D-DCT
def get_1d_dct(list_1d):
    coe_1d = np.zeros(list_1d.shape)
    N = list_1d.shape[0]
    for u in range(N):
        sum = 0.0
        C_u = 1/np.sqrt(2) if u == 0 else 1
        for x in range(N):
            sum += list_1d[x] * np.cos((2*x+1) * u * np.pi/(2*N))
        coe_1d[u] = sum * C_u * np.sqrt(2/N)

    return coe_1d


# Calculate 1D-IDCT
def get_1d_idct(coe_1d):
    list_1d = np.zeros_like(coe_1d)
    N = coe_1d.shape[0]
    for x in range(N):
        sum = 0.0
        for u in range(N):
            C_u = 1/np.sqrt(2) if u == 0 else 1
            sum += C_u * coe_1d[u] * np.cos((2*x + 1)*u*np.pi/(2*N))
        list_1d[x] = sum * np.sqrt(2/N)
    
    return list_1d
        

# Calculate PSNR
def get_psnr(expect, actual):
    mse = np.mean((expect - actual) ** 2)
    if mse == 0:
        return 100
    # Grayscale image with 8 bit
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel*max_pixel) / mse)
    return psnr

# Visualize on log domain
def visualize_log_domain(coe_image):
    # Clip values to avoid log(0)
    dct_coefficients_log = np.log1p(np.abs(coe_image))  # Using log1p for numerical stability
    plt.imshow(dct_coefficients_log, cmap='gray')
    plt.title("DCT Coefficients (Log Domain)")
    plt.colorbar()
    plt.show()


# Helper function
def start_calculate(time_log):
    fp = open(f"./results/{time_log}.txt", "w")
    start = time.process_time()
    return {
        "file": fp,
        "start": start
    }

def end_calculate(start, data, data_log, file):
    np.save(f"./results/{data_log}", data)
    end = time.process_time()
    file.write(f"DCT TIME: {end - start} s\n")

def extract_image(save_file_name, load_file_name):
    cv2.imwrite(f"./images/{save_file_name}.png", np.load(f"./results/{load_file_name}.npy"))

# MAIN

# Get image
image = get_grayscale_image("lena.png")

# Number of rows / columns
N = image.shape[0]

########################################### 1D-DCT

# file_name = "1D_DCT"
# start_package = start_calculate(file_name)

# dct_1d = np.zeros_like(image)
# for i in tqdm(range(N), desc="dct_1d_x"):
#     dct_1d[i, :] = get_1d_dct(image[i, :])

# for i in tqdm(range(N), desc="dct_1d_y"):
#     dct_1d[:, i] = get_1d_dct(dct_1d[:,i])

# end_calculate(start_package["start"], dct_1d, f"{file_name}_COE", start_package["file"] )

file_name_idct = "1D_IDCT"
start_package = start_calculate(file_name_idct)

coe_1d_dct = np.load(f"./results/1D_DCT_COE.npy")
idct_1d = np.zeros_like(coe_1d_dct)

# Apply inverse DCT on the DCT coefficients
for i in tqdm(range(N), desc="idct_1d_x"):
    idct_1d[i, :] = get_1d_idct(coe_1d_dct[i, :])  # Correct input here

for i in tqdm(range(N), desc="idct_1d_y"):
    idct_1d[:, i] = get_1d_idct(idct_1d[:, i])  # Continue with inverse DCT

end_calculate(start_package["start"], idct_1d, file_name_idct, start_package["file"])

# Save the image after processing
processed_image = process_data(idct_1d)  # Clip and convert to uint8
cv2.imwrite("./images/lena_grayscale_reconstructed.png", processed_image)


########################################### 1D-DCT

