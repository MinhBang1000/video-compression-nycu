```markdown
# Image Compression with Run-Length Encoding and DCT

This project implements JPEG-like image compression using an 8x8 block-based Discrete Cosine Transform (DCT) on the "lena.png" image. The compression includes quantization, zigzag scanning, run-length encoding, and decoding, followed by image reconstruction using the inverse DCT. The project concludes with a comparison of compression efficiency using two quantization tables.

## Prerequisites

Ensure you have Python installed on your machine. The required libraries can be installed as shown below.

### Required Libraries
- `numpy`
- `opencv-python`
- `scipy`
- `matplotlib`

Install all libraries using:
```bash
pip install numpy opencv-python scipy matplotlib
```

## Project Structure

The project contains the following primary files:

- **VC_HW4_313540015.ipynb**: Jupyter notebook containing the code for DCT-based image compression, quantization, zigzag scanning, run-length encoding/decoding, and reconstruction.
- **lena.png**: The grayscale image used for compression.
- **README.md**: Instructions and information about the project.

## Project Workflow

The project performs the following steps:

1. **Load Image and Convert to Grayscale**: Loads `lena.png` and converts it to grayscale for simpler processing.
   
2. **8x8 Block Division**: Divides the grayscale image into 8x8 non-overlapping blocks to localize frequency transformation.

3. **2D DCT Transformation**: Applies the 2D Discrete Cosine Transform to each block to convert spatial data into frequency components.

4. **Quantization**: Uses two separate quantization tables to compress DCT coefficients by reducing precision in high frequencies.

5. **Zigzag Scan and Run-Length Encoding**:
   - Applies zigzag scanning within each block to order coefficients from low to high frequency.
   - Performs run-length encoding on the ordered coefficients, focusing on consecutive zeros for compression.

6. **Run-Length Decoding and Reconstruction**:
   - Decodes the compressed data, applies inverse quantization, and reconstructs the original blocks using inverse DCT.
   - Recombines all blocks to reconstruct the compressed image.

7. **Compression Efficiency Analysis**:
   - Compares the encoded image sizes for both quantization tables.
   - Calculates and prints PSNR (Peak Signal-to-Noise Ratio) values to assess the quality of reconstructed images.

## Results and Analysis

### Compression Results

The notebook calculates the compressed image sizes (based on run-length encoded data) for both quantization tables. The lower the size, the higher the compression, though with potential quality loss.

### PSNR Analysis

The PSNR values are computed for each reconstructed image to quantify quality loss after compression. Higher PSNR indicates better quality retention.

## Usage

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook VC_HW4_313540015.ipynb
   ```
2. Follow the instructions in the notebook to execute each cell sequentially.
3. Observe the output images and encoded sizes, as well as PSNR values, for a comparison of both quantization tables.

## Notes

- The notebook includes detailed comments for each section, explaining the purpose and method used.
- For better performance, the `scipy.fftpack.dct` and `idct` functions are recommended over manual DCT/IDCT implementations.

---

By following the steps in this README and the Jupyter notebook, you should be able to understand and execute a JPEG-like image compression process.
``` 