# FastParallelKNN

## Table of Contents
1. [Overview](#overview)
2. [Setup](#setup)
   - [Core Dependencies](#core-dependencies)
   - [Parallel Dependencies](#parallel-dependencies)
   - [Additional Tools](#additional-tools)
   - [Step-by-Step Installation](#step-by-step-installation)
3. [Project Structure](#project-structure)
   - [Folder Breakdown](#folder-breakdown)
4. [Code Overview](#code-overview)
   - [Exact k-NN Implementations](#1-exact-k-nn-implementations)
   - [Approximate k-NN Implementations](#2-approximate-k-nn-implementations)
   - [Utility Functions](#3-utility-functions)
5. [Benchmarking and Testing](#benchmarking-and-testing)
   - [Julia Benchmarks](#julia-benchmarks)
   - [MATLAB Benchmarks](#matlab-benchmarks)
6. [Results](#results)
7. [Troubleshooting](#troubleshooting)

## Overview

**FastParallelKNN** is a high-performance k-Nearest Neighbors (k-NN) search library implemented in C, with additional testing and benchmarking scripts in Julia and MATLAB. The project offers both **exact** and **approximate** k-NN solutions, with versions implemented for **serial** and **parallel** processing using **OpenMP**, **OpenCilk**, and **Pthreads**.

The primary aim is to provide efficient, scalable solutions for k-NN search, enabling the handling of large datasets. The library is designed to allow users flexibility in choosing between exact and approximate methods, as well as the parallel framework best suited to their hardware.

## Setup

### Core Dependencies

The following packages and libraries are required to build and run the project:

- **GCC**: The GNU Compiler Collection for building the project, including OpenMP support.
- **OpenBLAS**: An optimized BLAS (Basic Linear Algebra Subprograms) library for fast matrix operations.
- **GSL**: The GNU Scientific Library for advanced numerical computations.
- **HDF5 Library**: To manage and read datasets in HDF5 format.

### Parallel Dependencies

If you intend to use the parallel k-NN versions, ensure you have the following:

- **OpenMP**: Enabled by default in most modern GCC versions. It provides shared-memory parallelism.
- **OpenCilk**: A task-parallel runtime system for C/C++. Follow installation instructions from [OpenCilk GitHub](https://github.com/OpenCilk).
- **Pthreads**: Standard POSIX threads, available on most UNIX/Linux distributions.

### Additional Tools

- **Julia**: For benchmarking and performance visualization. [Download Julia](https://julialang.org/downloads/).
- **MATLAB**: To run additional benchmark scripts (optional).

### Step-by-Step Installation

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

3. **Install OpenCilk** (for OpenCilk-based implementations):
   - Download and follow instructions from [OpenCilk's Installation Guide](https://github.com/OpenCilk/opencilk-project).

4. **Build the Project**:
   - For GCC:
     ```bash
     make -f Makefile.gcc all
     ```
   - For Clang with OpenCilk:
     ```bash
     make -f Makefile.clang all
     ```

### Running the Code

To run the different implementations, you can use:

- **Serial Exact k-NN**:
  ```bash
  ./knn_project
  ```
- **Parallel Exact k-NN (OpenMP)**:
  ```bash
  ./knn_project_openmp
  ```
- **Parallel Exact k-NN (OpenCilk)**:
  ```bash
  ./knn_project_clang
  ```
- **Parallel Exact k-NN (Pthreads)**:
  ```bash
  ./knn_project_pthread
  ```

## Project Structure

```
├── build                 # Compiled object files and binaries
│   ├── exact             # Object files for exact k-NN implementations
│   ├── main.o            # Main executable object file
│   ├── tests             # Object files for test functions
│   └── utils             # Object files for utility functions
├── data                  # Dataset files (HDF5 format)
├── include               # Header files for C code
│   ├── approximate       # Headers for approximate k-NN algorithms
│   ├── exact             # Headers for exact k-NN algorithms
│   ├── tests             # Headers for testing functions
│   └── utils             # Utility headers (I/O, distance calculations, etc.)
├── julia                 # Julia scripts for benchmarking
│   ├── knnAlgorithms     # Specific k-NN algorithm implementations in Julia
│   └── plotBenchmarks.jl # Script to plot benchmark results
├── knn_project           # Main executable for GCC builds
├── knn_project_clang     # Main executable for Clang/OpenCilk builds
├── Makefile.gcc          # Makefile for GCC builds
├── Makefile.clang        # Makefile for Clang/OpenCilk builds
├── matlab                # MATLAB script for k-NN benchmarks
├── README.md             # Project documentation
├── results               # Folder for benchmark results, logs, and plots
├── run_all_knn.sh        # Shell script to run all k-NN implementations
├── run_main_knn.sh       # Shell script to run the main executable
├── run_main_opencilk_knn.sh # Shell script for OpenCilk version
└── src                   # Source code for k-NN implementations
    ├── approximate       # Approximate k-NN source files
    ├── exact             # Exact k-NN source files
    ├── tests             # Test functions for k-NN algorithms
    └── utils             # Utility source files (I/O, memory, distance)
```

### Folder Breakdown

- **data**: Contains HDF5 dataset files for testing.
- **include**: Header files organized by functionality (exact/approximate, utility).
- **julia**: Julia scripts for additional testing and benchmarks.
- **matlab**: MATLAB scripts for comparing results.
- **results**: Output directory for logs, benchmarks, and plots.
- **src**: Core source files for the k-NN implementations (exact/approximate) and utilities.

## Code Overview

### 1. Exact k-NN Implementations

- **Serial Version**: Implements k-NN using a basic brute-force approach.
- **Parallel Versions**:
  - **OpenMP**: Leverages shared-memory parallelism.
  - **OpenCilk**: Utilizes task-based parallelism for dynamic load balancing.
  - **Pthreads**: Provides thread-level parallelism for granular control.

### 2. Approximate k-NN Implementations

- **Serial Version**: Implements approximate k-NN using probabilistic techniques like Locality-Sensitive Hashing (LSH).
- **Parallel Versions**: These implementations are parallelized to improve performance, with acceptable accuracy trade-offs.

### 3. Utility Functions

Utility functions handle crucial tasks like:

- **Dataset I/O**: Load and save data in HDF5 format.
- **Distance Calculations**: Perform efficient distance computations using matrix operations.
- **Memory Management**: Optimize dataset handling based on system memory.

## Benchmarking and Testing

### Julia Benchmarks

Julia scripts in `julia/knnAlgorithms` provide benchmarks for serial and parallel implementations. To run a Julia benchmark:
```bash
julia julia/knnAlgorithms/knnExactSerial.jl
```

Use `plotBenchmarks.jl` to visualize benchmark results.

### MATLAB Benchmarks

For MATLAB, use the script in the `matlab` folder:
```matlab
run('matlab/knnBenchmark.m')
```

## Results

The `results` directory holds:

- **Benchmarks**: Raw test performance data.
- **Logs**: Execution logs for debugging and analysis.
- **Plots**: Visualization of k-NN accuracy and speed.

## Troubleshooting

- **Compiling Errors**: Ensure dependencies are installed (`libhdf5`, `libopenblas`, `libgsl`).
- **Dataset Errors**: Confirm the HDF5 dataset files are correctly formatted and accessible.
- **OpenCilk Issues**: If `OpenCilk` is not recognized, check the installation path and update your `PATH` environment variable.
