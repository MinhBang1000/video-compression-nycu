# Image Channel Processing

This project processes an image to extract its RGB, YUV, and CbCr channels using specific formulas. The processed channels are then saved as grayscale images.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)

## Requirements

- Python 3.x
- NumPy
- OpenCV

## Installation

1. **Download the project ZIP file** from the provided source.

2. **Extract the ZIP file** to your desired directory.

3. **Navigate to the extracted project directory**.

4. **Install the required packages**:
   You can use pip to install the necessary packages. It is recommended to do this in a virtual environment.
   
   ```bash
   pip install numpy opencv-python
   ```

## Usage

1. **Place your image**: Ensure you have an image named `lena.png` in the same directory as the script.

2. **Run the script**:
   Execute the script using Python. This will read the image, process it, and save the resulting channel images in the `images` directory.
   
   ```bash
   python <script-name>.py
   ```

3. **View the processed images**: The output images will be saved in the `./images/` directory, with filenames corresponding to their respective channels (e.g., `R.png`, `G.png`, `B.png`, `Y.png`, `U.png`, `V.png`, `Cb.png`, `Cr.png`).

## Directory Structure

```
project-directory/
│
├── lena.png           # Input image file
├── <script-name>.py   # Python script for processing the image
└── images/            # Directory where processed channel images will be saved
    ├── R.png
    ├── G.png
    ├── B.png
    ├── Y.png
    ├── U.png
    ├── V.png
    ├── Cb.png
    └── Cr.png
```