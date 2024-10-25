# Motion Estimation and PSNR Calculation

This project implements motion estimation techniques and compares the predicted frames with actual frames using PSNR (Peak Signal-to-Noise Ratio). The script performs full search motion estimation, generates predicted and residual frames, and displays the output along with runtime statistics.

## Prerequisites

Ensure that you have Python installed on your machine. You also need to install the required libraries before running the notebook.

### Required Libraries

The following Python libraries are required:

- `numpy`
- `matplotlib`
- `time`

You can install these libraries by running the following command in your terminal:

```bash
pip install numpy matplotlib
```

## Instructions

Follow these steps to run the notebook in the correct order:

1. **Extract the project.**
   
   Download and extract the project (`VC_HW3_313540015.ipynb`). You can use the anaconda or any environment can run the jupyter notebook for Python code. In this case, I am using Anaconda Navigator [visit](https://anaconda.org/). If you use Anaconda, please create your own enviroment before launch the jupyter notebook or visual studio code to open my project. In my case, I am using vscode that was launched from Anaconda Navigator.

2. **Install required libraries.**

   Install the necessary libraries by running the following command:

   ```bash
   pip install numpy matplotlib
   ```

3. **Open the notebook.**

   Launch Jupyter Notebook or JupyterLab in the terminal by running (Can use different tools):

   ```bash
   jupyter notebook
   ```

   Then, open the notebook file `VC_HW3_313540015.ipynb`.

4. **Run the cells in order.**

   Ensure that you run each cell in sequence to generate the predicted and residual frames and calculate the PSNR and runtime statistics. Here are the key steps:
   - **Step 1:** Importing library.
   - **Step 2:** Load the frames and initialize variables (e.g., `num_blocks_x`, `num_blocks_y`, etc.).
   - **Step 3:** Define functions.
   - **Step 4:** For full search, for three step search, review the predicted frame and the psnr values.
   - **Step 5:** Compare the results of Full Search algorithm with different values of search ranges.

5. **Results.**

   The notebook will display the predicted and residual frames, along with the calculated PSNR and runtime statistics for analysis.

6. **Notice**
    We can change the block size. Just change it in the second step.
## Output

The script will generate and display the following:

- **Predicted Frame**: Visual representation of the predicted frame using motion estimation.
- **Residual Frame**: The difference between the actual and predicted frames.
- **PSNR Value**: The PSNR value for quality assessment.
- **Runtime**: The time taken to compute the motion estimation.