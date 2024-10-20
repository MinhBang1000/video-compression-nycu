# 1D and 2D DCT / IDCT, PSNR

This project demonstrates converting a digital image from the time domain to the frequency domain using **Discrete Cosine Transform (DCT)** for compression purposes. Additionally, the project reconstructs the image from frequency domain coefficient values using **Inverse Discrete Cosine Transform (IDCT)** and evaluates the **PSNR (Peak Signal-to-Noise Ratio)** to assess the quality of the reconstruction.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Important Notes](#important-notes)

## Requirements

- Python 3.x
- NumPy
- OpenCV
- Matplotlib
- tqdm

## Installation

1. **Download the project ZIP file** from the provided source.

2. **Extract the ZIP file** to your desired directory.

3. **Navigate to the extracted project directory**.

4. **Install the required packages**:
   Use `pip` to install all necessary packages. It is recommended to do this in a virtual environment.

   ```bash
   pip install numpy opencv-python matplotlib tqdm
   ```

## Usage

1. **Place your image**: Ensure you have an image named `lena.png` in the same directory as the script.

2. **Run the script**:
   Execute the script using Python. This will read the image, process it, and save the resulting files in the `results` directory. You need to manually create the `images` and `results` directories before running the script.

   ```bash
   python VC_HW2_313540015.py
   ```

   or 

   ```bash
   py VC_HW2_313540015.py
   ```

## Directory Structure

```
project-directory/
│
├── lena.png               # Input image file
├── VC_HW2_313540015.py    # Python script for processing the image
├── images/                # Directory where processed images (optional) can be saved
└── results/               # Directory where the result files will be saved
    ├── 1D_DCT_COE.npy               # Coefficient values for 1D DCT
    ├── 1D_DCT_COE_TIME_LOG.txt      # Processing time for 1D DCT
    ├── 1D_DCT_PSNR.txt              # PSNR value for 1D DCT reconstruction
    ├── 1D_IDCT.npy                  # Reconstructed image from 1D IDCT
    ├── 1D_IDCT_TIME_LOG.txt         # Processing time for 1D IDCT
    ├── 2D_DCT_COE.npy               # Coefficient values for 2D DCT
    ├── 2D_DCT_COE_TIME_LOG.txt      # Processing time for 2D DCT
    ├── 2D_DCT_PSNR.txt              # PSNR value for 2D DCT reconstruction
    ├── 2D_IDCT.npy                  # Reconstructed image from 2D IDCT
    └── 2D_IDCT_TIME_LOG.txt         # Processing time for 2D IDCT
```

## Important Notes

1. **Running the Script**:
    - **BLOCK 1**: This block should be run **every time** you execute the script. It handles the common setup and processing steps.
    
    - **BLOCK 2**: This block contains the code for running the **1D DCT/IDCT** process. To run it:
      - Uncomment **BLOCK 2** in the code and run the script.
      - The results for 1D DCT, 1D IDCT, and PSNR will be saved in the `results/` directory.
    
    - **BLOCK 3**: To run the **2D DCT/IDCT** process:
      - Comment out **BLOCK 2** and uncomment **BLOCK 3** in the code.
      - The results for 2D DCT, 2D IDCT, and PSNR will also be saved in the `results/` directory.

2. **Manual Directory Creation**:
    - Before running the script, you need to manually create the `images/` and `results/` directories, where the results and optional images will be stored.