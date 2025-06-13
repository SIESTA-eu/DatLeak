# UNDER DEVELOPMENT
When anonymizing data, for instance, by randomizing data orders, it's important to implement safeguards against potential data leakage. Data leakage can occur if scrambled variables inadvertently retain patterns that could be traced back to the original participants. Hence DatLeak can be run to test for data leakage. 

# Purpose
The purpose of this repository is to analyze information leakage in two different data types of **NeuroImaging** and **Tabular** datasets. Each of which consist of **Original** and **Scrambled/Synthetic** versions. 


## Table of Contents
- [Purpose](#purpose)
- [Tabular DataLeak](#tabular-dataleak)
  - [Method (Tabular)](#method-tabular)
  - [Usage (Tabular)](#usage-tabular)
  - [Output (Tabular)](#output-tabular)
- [NeuroImaging DataLeak](#neuroimaging-dataleak)
  - [Method (NeuroImaging)](#method-neuroimaging)
  - [Pseudocode](#pseudocode)
  - [Usage (NeuroImaging)](#usage-neuroimaging)
  - [Output (NeuroImaging)](#output-neuroimaging)
  - [Full/Partial Leakage Calculation](#fullpartial-leakage-calculation)
  - [Test](#test)
- [HTML report](#html-report)


# Tabular DataLeak
Methods for detection of data leakage in a tabular dataset.
## Method (Tabular)

### Full Leakage
A row/participant $i$ is considered to have **full leakage** if the number of matching cells equals the number of valid cells in that row. The condition for full leakage for row $i$ is:

```math
\text{Full Leakage for Row } i = 
\begin{cases}
1, & \text{if } \text{match\_count}_{i} = \text{valid\_mask\_sum}_{i} \\
0, & \text{otherwise}
\end{cases}
```
```math
\text{Full Leakage Percentage} = \left( \frac{\text{Full Leakage Count}}{\text{total\_rows}} \right) \times 100
```

### Partial Leakage

A row/participant $i$ is considered to have **partial leakage** if the number of matching cells is greater than 0 but less than the number of valid cells in that row. The condition for partial leakage for row `i` is:

```math
\text{Partial Leakage for Row } i = 
\begin{cases}
1, & \text{if } 0 < \text{match\_count}_{i} < \text{valid\_mask\_sum}_{i} \\
0, & \text{otherwise}
\end{cases}
```
```math
\text{Partial Leakage Percentage} = \left( \frac{\text{Partial Leakage Count}}{\text{total\_rows}} \right) \times 100
```

This formula checks if some, but not all, valid cells match between the original and scrambled rows, indicating partial leakage. 

The script detects data leakage in a tabular dataset by comparing an original with an anonymized version. It calculates percentages of full leakage (all variables are the same), and partial leakage (some variables are the same). In the latter case, it does so by averaging matching cells (per row). The script accepts command-line inputs for the dataset files (CSV or TSV) and an optional ignore value.

## Usage (Tabular)

```
python DatLeak.py <original_file> <scrambled_file> [ignore_value] [ignore_col]
```
- <original_file>: Path to the CSV or TSV file containing the original data.
- <scrambled_file>: Path to the CSV or TSV file containing the scrambled data.
- [ignore_value]: A value to ignore during the comparison (e.g., NaN or any placeholder value).
- [ignore_col]: A column to ignore during the comparison (ignore_col can be a single integer (e.g., 1) or comma-separated list (e.g., '1,2,3') or list literal (e.g., '[1,2,3]')).

### Example use

```
python DatLeak.py test_files/data_original.tsv test_files/data_scramble.tsv -999 0
# or
python DatLeak.py test_files/data_original.tsv test_files/data_scramble.tsv None 0
# even
python DatLeak.py test_files/data_original.tsv test_files/data_scramble.tsv
```


## Output (Tabular) 

```
Partial Leakage: 99.78%
Full Leakage: 0.00%
Average Matching Cells per Row: 4.98
Standard Deviation of Matching Cells per Row: 1.68
```
- Partial Leakage: The percentage of rows with partial leakage.
- Full Leakage: The percentage of rows with full leakage.
- Average Matching cells per row.
- Standard Deviation of matching cells per row.

# NeuroImaging DataLeak
DataLeakage analysis in NeuroImaging data

**Supported BIDs NeuroImaging data type:**
- 3D structural T1-weighted fMRI
- 4D fMRI image
- MEG (Magnetoencephalography)
- EEG (Electroencephalography)
  
## Method (NeuroImaging)
We use 3 known methods to measure the similarity between two versions by quantifying information leakage across all dimensions of dataset.

The idea of using three methods is to quantify information leakage where each is to complement each other’s strengths and limitations, providing a more robust and comprehensive assessment of potential leakage between the original and scrambled data.

`NOTE` Due to the computationally intensive nature of comparing large 3D to 4D arrays slice by slice, we implemented SSIM and Pearson correlation manually instead of using built-in functions from libraries like scipy. To optimize performance, we use [`numba`](https://pypi.org/project/numba/) to JIT compile the function. However, numba does not support external Python callables like scipy.stats.pearsonr or skimage.ssim.
- [`Pearson:`](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) Pearson correlation coefficients
    - Useful for detecting global similarity patterns. However sensitive to scaling and outliers, but effective when relationships are purely linear.
- [`SSIM:`](https://en.wikipedia.org/wiki/Structural_similarity_index_measure) Structural similarity index measure
    - SSIM is designed to measure perceptual similarity, which takes into account luminance, contrast, and structure.
- [`np.allclose:`](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html) Boolean matrix showing exact match (leakage) in centered data
    - A strict comparison to identify exact matches, indicating serious leakage. It also helps to detect accidental duplication or information leakage due to transformations.
### Pseudocode
### 3D
```terminal
# Given a dimension in [x, y, z]
For slice i=0 to i=n-1: # for each original image slice
    For j=0 to j=n-1:   # comparing all slices of scrambled 
    a. Extract 2D slices from original and scrambled based on axis:
        slice_o ← Original[plane/2D slice] 
        slice_s ← Scrambled[plane/2D slice] 
    b. Full leakage check:
        f_l_corrs ← np.allclose(slice_o, slice_s)
    c. Compute Pearson correlation:
        p_corrs ← p_corr(slice_o, slice_s)
    d. Compute SSIM score:
        s_corrs ← ssim(slice_o, slice_s)
return a numpy array of computed in shape (x,x), (y,y) or (z,z)
e. Calculate Full/Partial leakage
```
### 4D & 2D
```terminal
# Method 1 [Spatial Analysis]
For time t=0 to t=n-1: 
    a. compute as 3D as above
return mean(3D)

# Method 2 [Temporal Analysis]
a. Extract 1 array of voxels along time dimension in either origianl or scrambled
b. Calculate Pearson corr, SSIM, np.allclose
c. Calculate Full/Partial leakage
```
## Usage (NeuroImaging)
```terminal
python run.py <Original Base Dir> <Scrambled Base Dir> [report]
```
- Original/Scrambled base directory: The function searches for images, given one directory above all images
- Report is optional. Takes two arguments of True/False. By default is False
### Example
```terminal
python run.py "usecase-2.2/input" "usecase-2.2/scrambled" False
```
## Output (NeuroImaging)
```terminal
- Image Info
- Spatial Analysis:
    - dim[x]  Full Leakage: 0/n voxels   Partial Leakage: leakage value
    - dim[y]  Full Leakage: 0/n voxels   Partial Leakage: leakage value
    - dim[z]  Full Leakage: 0/n voxels   Partial Leakage: leakage value
- Temporal Analysis[if applicable]:
    - Full Leakage: 0/n voxels   Partial Leakage: leakage value
- Partial Leakage: numerical value
- Full Leakage: True/False
```
## Full/Partial Leakage Calculation
### `Full Leakage:` 
We consider Full Leakage as exact identical array of voxels, and/or perfect linear relationship between voxels. Meaning if an original image is 100% identical to the scrambled version. If Pearson Correlation returns a value of [0.99999:1.0], backed up by np.allclose which returns similar value in any slice across any dimension of [x, y, z].
### `Partial Leakage:` 
Partial Leakage is calculated from the distribution of **max values** extracted from the Pearson correlation matrices.

Maximum linear relation between one array of voxels in original image compared to all arrays of voxels in the scrambled version, identifies how much information is leaked. Therefore we focus only on the **max values**. The below example matrix shows an array of correlations value of shape (x, x) reduces to (x, ) dimension, keeping only max values. [TO BE EDITED] 

```math
\left.
  \begin{matrix}
    x_{o1,s1} & \dots & x_{o1,sN} \\
    x_{o2,s1} & \dots & x_{o2,sN} \\
    \vdots    & \ddots & \vdots  \\
    x_{oN,s1} & \dots & x_{oN,sN} \\
  \end{matrix}
\right\}
\quad
\left.
  \begin{matrix}
    x_{\text{max}1} \\
    x_{\text{max}2} \\
    \vdots \\
    x_{\text{max}N}
  \end{matrix}
\right.
```
## Test
We provide a test case in Jupyter Notebook files, where an image is downloaded and different test cases are evaluated. The notebooks can be found in **NeuroImaging/test/** folders. We elaborate further what would be the expectation after model evaluation.   

## HTML Report
To ensure transparency and verification, and allow for visual inspection of the results, an optional HTML report can be generated alongside the leakage analysis. The report provides:

- A side-by-side visual snapshot of the original and scrambled data of different dimensions.
    - In case of MEG/EEG files, the plots will be Power Spectral Density (PSD)
- A summary table displaying the computed values for:
    - Pearson correlation: min, max, mean along [x, y, z] dimension
    - SSIM: min, max, mean along [x, y, z] dimension
- Plots visualizing the distribution of correlation values, helping to identify any unexpected alignment or leakage patterns.
    - Total distribution plots of correlations
    - Distribution plots of maximum values of potential leakage.
    - Identical voxels plot
    - Correlation distribution along time dimension. If applicable to time 
Note: HTML report targets only NeuroImaging data.
