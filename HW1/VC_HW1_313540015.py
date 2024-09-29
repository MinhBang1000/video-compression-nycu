import numpy as np
import cv2

# Read the image by OpenCV
image = cv2.imread("lena.png")

# Assign to R,G,B matrix
R = image[:,:,0]
G = image[:,:,1]
B = image[:,:,2]

# Find YUV and CbCr by our provided formula
Y = 0.299 * R + 0.587 * G + 0.114 * B
U = -0.169 * R - 0.331 * G + 0.5 * B + 128
V = 0.5 * R - 0.419 * G - 0.081 * B + 128
Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128

# Process channels
def process_data(channel):
    channel = np.clip(channel, 0, 255) # Make values in range from 0 to 255 
    return channel.astype(np.uint8) # Change value type to uint8

# Summary all channels
channels = {
    "R": process_data(R),
    "G": process_data(G),
    "B": process_data(B),
    "Y": process_data(Y),
    "U": process_data(U),
    "V": process_data(V),
    "Cb": process_data(Cb),
    "Cr": process_data(Cr)
}

# Create an grayscale image for each channel
for key, value in channels.items():
    cv2.imwrite(f"./images/{key}.png", value)


