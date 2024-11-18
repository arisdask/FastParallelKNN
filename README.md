# FastParallelKNN

## Overview

**FastParallelKNN** is a high-performance k-Nearest Neighbors (k-NN) search library implemented in C, with bindings and benchmarks in Julia and MATLAB. The project includes both **exact** and **approximate** k-NN algorithms, with support for **serial** and **parallel** implementations using **OpenMP**, **OpenCilk**, and **Pthreads**.

The goal is to deliver efficient and scalable k-NN solutions that can handle large datasets. The code is designed with flexibility in mind, allowing users to switch between exact and approximate methods, and choose the parallel framework that best fits their hardware environment.

## Project Structure

```
├── data                 # Data files (e.g., HDF5 datasets)
├── include              # Header files
│   ├── approximate      # Approximate k-NN headers
│   ├── exact            # Exact k-NN headers
│   └── utils            # Utility headers (I/O, memory, etc.)
├── julia                # Julia code for testing and benchmarking
├── matlab               # MATLAB code for benchmarking
├── results              # Output files for benchmarks, logs, and plots
├── src                  # Source code (C implementations)
│   ├── approximate      # Approximate k-NN C implementations
│   ├── exact            # Exact k-NN C implementations
│   └── utils            # Utility functions
└── README.md            # This file
```

## Dependencies

### Core Dependencies

The following libraries and tools are required to build and run the project:

- **GCC**: GNU Compiler Collection, with support for OpenMP.
- **OpenBLAS**: Optimized BLAS (Basic Linear Algebra Subprograms) library.
- **GSL**: GNU Scientific Library for numerical computations.
- **HDF5 Library**: For handling and loading datasets in HDF5 format.

### Parallel Dependencies

If you plan to run parallel versions of the k-NN search, make sure you have the following tools installed:

- **OpenMP**: Supported by most modern GCC versions.
- **OpenCilk**: For task-parallel implementations. Installation instructions can be found at [OpenCilk GitHub](https://github.com/OpenCilk).
- **Pthreads**: Standard threading library available in UNIX/Linux environments.

### Additional Tools

- **Julia**: For benchmarking and plotting performance data. Installation instructions can be found at [JuliaLang](https://julialang.org/downloads/).
- **MATLAB**: For running additional benchmark scripts.

## Building the Project

### Step-by-Step Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/FastParallelKNN.git
   cd FastParallelKNN
   ```

2. **Install Core Dependencies**:
   - On Ubuntu:
     ```bash
     sudo apt-get update
     sudo apt-get install gcc libopenblas-dev libgsl-dev libhdf5-dev
     ```

3. **Build the Project**:
   - To compile all implementations (serial, approximate, and parallel):
     ```bash
     make all
     ```
   - This will generate the executables in the root directory.

### Running the Code

After building the project, you can run each implementation based on your requirements:

- **Serial Exact k-NN**:
  ```bash
  ./knn_exact_serial
  ```
- **Parallel Exact k-NN (OpenMP)**:
  ```bash
  ./knn_exact_openmp
  ```
- **Parallel Exact k-NN (OpenCilk)**:
  ```bash
  ./knn_exact_opencilk
  ```
- **Parallel Exact k-NN (Pthreads)**:
  ```bash
  ./knn_exact_pthread
  ```

For the approximate implementations, similar commands apply:

- **Serial Approximate k-NN**:
  ```bash
  ./knn_approx_serial
  ```
- **Parallel Approximate k-NN (OpenMP)**:
  ```bash
  ./knn_approx_openmp
  ```
- **Parallel Approximate k-NN (OpenCilk)**:
  ```bash
  ./knn_approx_opencilk
  ```
- **Parallel Approximate k-NN (Pthreads)**:
  ```bash
  ./knn_approx_pthread
  ```

## Code Overview

### 1. Exact k-NN Implementations

The exact k-NN implementations find the exact k nearest neighbors using Euclidean distance:

- **Serial Version**: A straightforward implementation of the k-NN algorithm, iterating over all data points.
- **Parallel Versions**:
  - **OpenMP**: Uses shared-memory parallelism to distribute distance calculations across multiple threads.
  - **OpenCilk**: A task-based parallel model to handle load balancing efficiently.
  - **Pthreads**: A low-level threading model for fine control over thread management.

### 2. Approximate k-NN Implementations

The approximate k-NN versions aim to reduce computation time by trading a small amount of accuracy:

- **Serial Version**: A baseline implementation using probabilistic and space-partitioning techniques like Locality-Sensitive Hashing (LSH).
- **Parallel Versions**: These versions leverage different parallel frameworks to enhance performance while maintaining reasonable accuracy.

### 3. Utility Functions

Utility functions handle core operations such as:

- **Dataset I/O**: Reading and writing datasets in HDF5 format.
- **Distance Calculations**: Computing squared Euclidean distances between points.
- **Memory Management**: Checking available memory to optimize dataset handling.

## Benchmarking and Testing

### Julia Benchmarks

The `julia/knnAlgorithms` folder contains Julia scripts for benchmarking the performance of different k-NN implementations:

- **Serial and Parallel Benchmarks**: Measure execution time, memory usage, and accuracy across exact and approximate methods.
- **Plotting**: Use the `plotBenchmarks.jl` script to generate visual plots of the benchmarking results.

To run the Julia benchmarks, ensure you have Julia installed and execute:
```bash
julia julia/knnAlgorithms/knnExactSerial.jl
```

### MATLAB Benchmarks

In the `matlab` folder, there is a MATLAB script to validate the k-NN implementations against existing MATLAB k-NN functions:
```matlab
run('matlab/knnBenchmark.m')
```

## Results

All benchmarking results, logs, and plots will be saved in the `results` folder:

- **Benchmarks**: Raw performance data from tests.
- **Logs**: Detailed logs of the k-NN executions.
- **Plots**: Graphical analysis of performance across implementations.

## Troubleshooting

- **Linking Errors**: Ensure the HDF5 library and other dependencies are correctly linked. The `Makefile` includes flags like `-lhdf5` and `$(pkg-config --cflags --libs hdf5)` to help with proper linking.
- **Dataset Errors**: Make sure that the dataset paths are correct and that the files are formatted properly in HDF5.
