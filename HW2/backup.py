import numpy as np
import cv2

# 1. convert to grayscale image
# Read the image by OpenCV
image = cv2.imread("lena.png")

# Assign to R,G,B matrix
R = image[:,:,0]
G = image[:,:,1]
B = image[:,:,2]


# Find YUV and CbCr by our provided formula
Y = 0.299 * R + 0.587 * G + 0.114 * B

# Process channels
def process_data(channel):
    channel = np.clip(channel, 0, 255) # Make values in range from 0 to 255 
    return channel.astype(np.uint8) # Change value type to uint8

# Create an grayscale image for Y channel
# cv2.imwrite(f"./images/lena_Y_grayscale.png", process_data(Y))
# print("Grayscale Y channel image saved successfully.")

# This image has 512x512 pixels
# print(image.shape)


# Some helper function
fp = open('report_2d.txt', 'w')

start = time.process_time()
dct_2d = np.zeros(image.shape)


end = time.process_time()

np.save('dct_2d', dct_2d)
fp.write(f"dct time: {end - start} s\n")