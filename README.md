# FastParallelKNN

## Table of Contents
1. [Overview](#overview)
2. [Setup](#setup)
   - [Core Dependencies](#core-dependencies)
   - [Parallel Dependencies](#parallel-dependencies)
   - [Additional Tools](#additional-tools)
   - [Step-by-Step Installation](#step-by-step-installation)
3. [Code Overview](#code-overview)
4. [Memory Management](#memory-management)
5. [Build and Run Project with `.sh` Script](#build-and-run-project-with-sh-script)
6. [Troubleshooting](#troubleshooting)
7. [Timestamps and PC Specs](#timestamps-and-pc-specs)


## Overview

**FastParallelKNN** is a high-performance library for solving the $k$-Nearest Neighbors ($k$-NN) problem, implemented in C with additional tools for testing and benchmarking in Julia. The library includes both **exact** and **approximate** $k$-NN algorithms, with implementations for **serial** and **parallel** processing using **OpenMP**, **OpenCilk**, and **Pthreads**.

The primary focus is to efficiently solve the all-to-all-$k$-NN problem, where each data point in a dataset is compared to all other data points to identify the $k$ nearest neighbors. Mathematically, given a set of points $C$ (corpus) and a query set $Q$, the $k$-NN finds for each point in $Q$ the $k$ closest points in $C$. If $C == Q$ then we have an *all-to-all*-$k$-NN problem. 

For large datasets, calculating all-to-all-$k$-NN is not time efficient - especially when our machine's available memory/RAM is very limited. To encounter this issue, instead of just creating functions (both with serial and parallel programming approach) which calculate accurately the exact solution, we implement functions which approximate solve the all-to-all-$k$-NN problem. 

The project also provides a generic `knnsearch` function (*core function*) comparable to MATLAB's `[idx, dst]= knnsearch(C, Q, 'k', k)`.


## Setup

### Core Dependencies

The following libraries are required to build and run the project:

- **GCC**: The GNU Compiler Collection, which includes OpenMP for parallelization.
- **OpenBLAS**: An optimized BLAS (Basic Linear Algebra Subprograms) library for efficient matrix operations.
- **GSL**: The GNU Scientific Library for advanced numerical computations.
- **HDF5 Library**: For handling datasets stored in HDF5 format.

### Parallel Dependencies

If you want to use the parallel $k$-NN implementations, ensure the following:

- **OpenMP**: A parallel programming interface included in most modern GCC versions.
- **OpenCilk**: A task-parallel runtime system for C/C++. Follow the [OpenCilk GitHub installation guide](https://github.com/OpenCilk/opencilk-project) for setup.
- **Pthreads**: Standard POSIX threads, available on most UNIX/Linux systems.

### Additional Tools

- **Julia**: For the initial algorithm design - benchmarking and visualizations. [Download Julia](https://julialang.org/downloads/).

### Step-by-Step Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/arisdask/FastParallelKNN.git
   cd FastParallelKNN
   ```

2. **Install Core Dependencies**:
   - On Ubuntu:
     ```bash
     sudo apt-get update
     sudo apt-get install gcc libopenblas-dev libgsl-dev libhdf5-dev
     ```

3. **Install OpenCilk** (for OpenCilk-based implementations):
   - Follow [OpenCilk's Installation Guide](https://github.com/OpenCilk/opencilk-project).
   - Recommended installation steps:
     1. Visit [OpenCilk Installation](https://www.opencilk.org/doc/users-guide/install/).
     2. Download the appropriate `.sh` file for your OS and run the extraction command.
     3. Move the extracted directory to `/opt/OpenCilk` so that `/opt/OpenCilk/bin` contains the `clang` executable.
     4. Run:
        ```bash
        export PATH="/opt/OpenCilk/bin/:$PATH"
        clang --version
        ```
        Check that the version output corresponds to OpenCilk:
        ```
        clang version 16.0.6 (OpenCilk)
        Target: x86_64-unknown-linux-gnu
        ```
     5. If necessary, adjust the `Makefile.clang` to point to the correct `clang` path.

4. **Build the Project**:
   - For GCC (excluding the OpenCilk implementations):
     ```bash
     make -f Makefile.gcc clean
     make -f Makefile.gcc
     ```
   - For Clang (including only OpenCilk functions):
     ```bash
     make -f Makefile.clang clean
     make -f Makefile.clang
     ```

### Running the Code

***Info: Before building and running the code, read the [Memory Management](#memory-management) section to verify that you set correctly your available memory inside the `mem_info.h` file (and re-build if needed).***

#### **Executable Files**
The executable files are `./knn_project` and `./knn_project_clang`, depending on the Makefile used to build the project. Both executables accept the same input format:

```
./[knn_project or knn_project_clang] [method] [num_of_threads] [data_path] [corpus_name] [query_name] [k] [compare_results (optional)] [neighbors (optional)] [distances (optional)]
```

#### **Input Arguments Explanation**
1. **method**: Specifies which implementation of the k-NN search you want to run (exact or approximate). Refer to the method ID table for the correct values.
   
2. **num_of_threads**: The number of threads you want to use for parallel execution. This is ignored for serial implementations.

3. **data_path**: The path to the `.hdf5` file that contains the datasets. The program will open this file to read the specified datasets.

4. **corpus_name**: The name of the dataset within the `.hdf5` file that represents the corpus.

5. **query_name**: The name of the dataset within the `.hdf5` file that represents the query set. If the query set is the same as the corpus, this should be the same as `corpus_name`.

6. **k**: Specifies the number of nearest neighbors (`k-NN`) you want to find.

7. **compare_results** (optional): The path to an `.hdf5` file that contains the expected results for comparison. This file must contain two datasets: one for `neighbors` and one for `distances`.

8. **neighbors** (optional): The name of the dataset in the `compare_results` file that holds the expected nearest neighbors to compare against your program's results.

9. **distances** (optional): The name of the dataset in the `compare_results` file that holds the expected distances to the nearest neighbors for comparison.

| method | id  |
|---|---|
|  Run all the *exact* knn functions and evaluate/compare the results based on the given dataset |  0 |
| Run all the *approx* knn functions and evaluate/compare the results (based on the exact results of an exact knn). The approx solutions solve only the all-to-all k-NN (C == Q) |  1 |
| Random Data Test for knn_approx_pthread (Playground) |  2 |
| Add your own custom tests here (we already have extra tests for the approximate methods using the sift-128-euclidean.hdf5 dataset)|  3 |

- **For the OpenCilk we need to set the `CILK_NWORKERS` beforehand**:
  ```bash
  export CILK_NWORKERS=[num_of_threads]
  ./knn_project_clang ...
  ```

To automate the build and execution with different methods and thread counts, **use the provided shell scripts**. See the [Script Section](#build-and-run-project-with-sh-script) below.


## Code Overview
*For the mathematical explanation of each method check the report.pdf*

### 1. Exact k-NN Implementations

- **Serial Version**: A basic brute-force approach for k-NN computation.
- **Parallel Versions**:
  - **OpenMP**: Uses shared-memory parallelism for faster computation.
  - **OpenCilk**: Employs task-based parallelism for dynamic load balancing.
  - **Pthreads**: Implements thread-level parallelism for fine control.

### 2. Approximate k-NN Implementations

- **Serial Version**: Implements approximate all-to-all k-NN using techniques explained in the report.pdf.
- **Parallel Versions**: Parallelized for better performance.

### 3. Utility Functions

Utility functions perform essential tasks:

- **Dataset I/O**: Manage loading of HDF5 data.
- **Distance Calculations**: Efficient computation of distances using OpenBLAS.
- [**Memory Management**](#memory-management): Adjust memory allocation based on your system specifications.


## Memory Management

The `mem_info.h` header plays a crucial role in managing memory usage throughout the project. Given that k-NN operations can be memory-intensive, especially with large datasets, efficient memory management is essential. This header provides functionality to query the system's available memory, allowing the program to dynamically or 'manually' adjust its operations based on current memory constraints.

### Key Features

1. **`size_t get_usable_memory(void)`**:
   - **Purpose**: Retrieves the current usable memory of the system, considering the overall available memory. This function ensures that the program doesn't consume all system resources, leaving enough headroom for other processes (and other threads) to run.
   - **Usage**: Utilized in the $k$-NN implementations to decide the maximum chunk size of data that can be processed without causing memory overflow.
   - **Implementation Note**: The function reads from `/proc/meminfo` (on Linux systems) to gather memory information (`MemAvailable`). It then calculates usable memory with a safety margin - `MEMORY_USAGE_RATIO` = $0.3$ ($30\%$), to prevent the system from running out of resources. 
   
   *Warning*: If still with the $30\%$ ratio too much memory is consumed, consider lower this ratio which is defined in the `mem_info.h`file

2. **`USE_CONST_MEMORY`**:
   - **Purpose**:
      - If $USE\_CONST\_MEMORY == 0$: 
        
        `get_usable_memory` returns the usable memory based on the available memory value inside the `/proc/meminfo` file (Linux Specific).
      - If $USE\_CONST\_MEMORY == 1$: 
        
        `get_usable_memory` always returns a constant prediction of the usable memory, `USABLE_MEM_PREDICTION`.

    ***Set:*** `USE_CONST_MEMORY` == 1 and `USABLE_MEM_PREDICTION` == [available **Kilobytes** prediction based on the system's specs]. A safe but still efficient prediction is to choose: $(3/8) * [Your\ System's\ Total\ RAM]$ when you run the exact functions and $(3/16) * [Your\ System's\ Total\ RAM]$ when you run the approximate ones (for no more than 8 threads and large datasets). In case you want to test for more number of threads consider to lower the value even more.  

    ***Warning:*** The computer which was used throughout the development of the project had $16GB$ of total RAM, so there is `USE_CONST_MEMORY` == 1 and `USABLE_MEM_PREDICTION` == 3000000, to run the approximate functions (the exact function can run also with 6000000). Change this value according to your own system's specs (and according to whether you run the exact or the approximate implementations). It is advised to start with smaller values and observe how close to fail the memory allocation is. 

### Why It Matters

The memory management provided by `mem_info.h` is essential for handling large-scale datasets. By setting how much memory is should be used per method, the program can decide how much data to load and process in chunks, which avoids exceeding memory limits and allows for efficient batch processing. This functionality is particularly valuable for parallel implementations where multiple threads or tasks need to share resources.

Not allowing to change this memory value is not ideal because running the process in multiple threads means each thread will have distinct memory requirements. Assuming that allocating the maximum possible memory to each thread would yield the optimal solution is overly simplistic. Instead, manually adjusting this value across different settings allows us to identify the optimal configuration for our specific system.

### Practical Impact

If the program crashes with the current number of threads, consider **reducing the `USABLE_MEM_PREDICTION` value**. This adjustment ensures that the memory usage remains within safe limits for the hardware configuration. On the other hand, if you aim to achieve better performance and have sufficient memory resources, you can **increase the `USABLE_MEM_PREDICTION` value** to allow the program to utilize more memory efficiently. (See comments to each test case in `main.c` to set the value correctly).



## Build and Run Project with `.sh` Script

***Warning: Make sure that the datasets we ask to call are actually stored inside the `/data` folder, otherwise they need to be downloaded/stored there - or you can change the filepath and the dataset names to call your own.*** (You can find the `sift-128-euclidean.hdf5` inside this [repo](https://github.com/erikbern/ann-benchmarks))

The `run_knn.sh` shell script is designed to facilitate running the `knn_project` executable by handling user inputs and constructing the command accordingly. Here are some key details to include:

#### **Usage of `run_knn.sh` Script**
1. **Purpose**: The script is a wrapper around the `knn_project` and `knn_project_clang` executables, allowing users to specify the k-NN search parameters easily from the command line.
  
2. **Execution Format**: The basic usage format for the script is:
   
   ```
   ./run_knn.sh [method] [num_of_threads] [data_path] [corpus_name] [query_name] [k] [compare_results (optional)] [neighbors (optional)] [distances (optional)]
   ```
   
   The script will determine whether to use `knn_project` or `knn_project_clang` based on the selected method.

3. **Method/Test Selection**: The script automatically decides which executable(s) to run based on the `method` value:
   
   - If `method` is `0`: Built and Run all the **exact** knn functions and evaluate/compare the results based on the given dataset
   - If `method` is `1`: Built and Run all the **approx** knn functions and evaluate/compare the results (based on the exact results of an exact knn). Keep in mind that the approximate solutions solve only the all-to-all k-NN problem in which $C == Q$. (*This test and test `2` are just 'playground' tests and they do not highlight the approximate method's advantages*).
   - If `method` is `3`: Built and Run all the **approx** knn functions for the `sift-128-euclidean.hdf5` having as corpus = query the train dataset ($10^6$ x $128$) and evaluate/compare the results based on an exact method. (The results of this test are presented in the [Timestamps and PC Specs](#timestamps-and-pc-specs) section)
   - Check the comments in `run_knn.sh` for more info

4. **Script Structure**:
   
   - **Command Construction**: Based on the user inputs, it constructs the appropriate command.
   - **Execution**: The constructed command is displayed and then executed.
   
5. **Example Commands**:
   
   - **k-NN exact functions test**:
     
     ```
     ./run_knn.sh 0 5 data/sift-128-euclidean.hdf5 train test 100 data/sift-128-euclidean.hdf5 neighbors distances
     ```
   
   - **k-NN approximate functions test**:
     
     ```
     ./run_knn.sh 1 4 null null null 100
     ./run_knn.sh 3 6 null null null 100
     ```

6. **Dependencies**: The script relies on correct installation of dependencies like OpenBLAS, GSL, HDF5, and OpenCilk for building and running the executables. *Make sure these are installed and configured properly.*

7. **Output**: The script will display the constructed command before execution.

## Troubleshooting

- **Compilation Errors**: Ensure all dependencies are correctly installed (`libhdf5`, `libopenblas`, `libgsl`).
- **Dataset Issues**: Check that HDF5 datasets are formatted and accessible.
- **OpenCilk Problems**: Ensure the correct `clang` version is installed and available in the system's PATH.
- **Precess Killed**: Consider lowering the value of `USABLE_MEM_PREDICTION` (see [**Memory Management**](#memory-management))

## Timestamps and PC Specs

Here are some basic spec info of the computer in which this project run: 

CPU:
- Architecture:             x86_64
  - CPU op-mode(s):         32-bit, 64-bit
  - Address sizes:          48 bits physical, 48 bits virtual
  - Byte Order:             Little Endian
- CPU(s):                   8
  - On-line CPU(s) list:    0-7
- Vendor ID:                AuthenticAMD
  - Model name:             AMD Ryzen 7 4700U with Radeon Graphics
    - CPU family:           23
    - Model:                96
    - Thread(s) per core:   1
    - Core(s) per socket:   8
    - Socket(s):            1
    - Stepping:             1
    - Frequency boost:      enabled
    - CPU max MHz:          2000,0000
    - CPU min MHz:          1400,0000
    - BogoMIPS:             3992.47
- Vendor ID:                AuthenticAMD
  - Model name:             AMD Ryzen 7 4700U with Radeon Graphics
- Caches (sum of all):      
  - L1d:                    256 KiB (8 instances)
  - L1i:                    256 KiB (8 instances)
  - L2:                     4 MiB (8 instances)
  - L3:                     8 MiB (2 instances)

RAM:
- MemTotal:       15715508 kB
---

We download the `sift-128-euclidean.hdf5` inside the data/ folder and run the test `3` for different number of threads. **Ensure** before running the next commands that you have correctly set the `USABLE_MEM_PREDICTION` value inside the `mem_info.h` according to the [**Memory Management**](#memory-management) section. In case of error see [Troubleshooting](#troubleshooting).

**Note:** It is not necessary to run `knn_exact_pthread` for every number of threads (we only run it once to obtain the correct exact results) nor to execute `knn_approx_serial` each time. However, ensure that both are executed at least during the first run; after that, you can comment them out within test case 3 in `main.c`.
```
./run_knn.sh 3 4 null null null 100
./run_knn.sh 3 6 null null null 100
./run_knn.sh 3 8 null null null 100
./run_knn.sh 3 12 null null null 100
```

Below is a table summarizing the results of the executions above, comparing the performance of the approximate parallel methods (`knn_approx_pthread`, `knn_approx_openmp`, and `knn_approx_opencilk`) with varying thread counts. Each row shows the runtime, query rates, and comparison metrics:

| **Threads** | **Method**               | **Running Time (s)** | **Queries per Second** | **Neighbors Hit Rate (%)** | **k-NN Avg Distance Rate (approx/exact)** |
|-------------|--------------------------|-----------------------|-------------------------|----------------------------|-------------------------------------------|
| **6**       | `knn_exact_pthread`      | 2589.54                | 386.17                  | -                          | -                                         |
|             | `knn_approx_serial`      | 1061.05                | 942.46                  | 99.90                      | 1.00023                                    |
| **4**       | `knn_approx_pthread`     | 414.91                 | 2410.16                 | 30.16                      | 1.06679                                    |
|             | `knn_approx_openmp`      | 420.97                 | 2375.47                 | 30.16                      | 1.06679                                    |
|             | `knn_approx_opencilk`    | 417.37                 | 2395.96                 | 30.16                      | 1.06679                                    |
| **6**       | `knn_approx_pthread`     | 294.48                 | 3395.80                 | 21.91                      | 1.08778                                    |
|             | `knn_approx_openmp`      | 284.41                 | 3516.05                 | 21.91                      | 1.08778                                    |
|             | `knn_approx_opencilk`    | 278.20                 | 3594.56                 | 21.91                      | 1.08778                                    |
| **8**       | `knn_approx_pthread`     | 219.75                 | 4550.52                 | 17.77                      | 1.10347                                    |
|             | `knn_approx_openmp`      | 226.39                 | 4417.21                 | 17.77                      | 1.10347                                    |
|             | `knn_approx_opencilk`    | 235.59                 | 4244.73                 | 17.77                      | 1.10347                                    |
| **12**      | `knn_approx_pthread`     | 142.66                 | 7009.78                 | 13.49                      | 1.12639                                    |
|             | `knn_approx_openmp`      | 171.12                 | 5843.99                 | 13.49                      | 1.12639                                    |
|             | `knn_approx_opencilk`    | 161.91                 | 6176.14                 | 13.49                      | 1.12639                                    |

### Observations

1. **Performance Across Thread Counts**:
   - Increasing the number of threads results in lower running times and higher query rates, showcasing the benefits of parallelization.
   - Accuracy, as represented by the `Neighbors Hit Rate` (and `k-NN Average Distances Rate`), decreases with more threads, reflecting a potential trade-off between speed and precision.

2. **Comparison of Methods**:
   - All approximate parallel methods (`knn_approx_pthread`, `knn_approx_openmp`, and `knn_approx_opencilk`) show consistent `Neighbors Hit Rate` and `k-NN Average Distances Rate` at each thread level.

3. **Theoretical Analysis**
    - While an accuracy of 17.77% for 8 threads may initially seem very low, it is important to consider the time savings achieved by the approximate methods. For instance, achieving 17.77% accuracy using the exact method would require running the program for approximately:
    $2589.54 \, \text{seconds (exact runtime)} \times 0.1777 \approx 460 \, \text{seconds}$. 
    In contrast, the approximate methods achieve the same accuracy in approximately half the time. This demonstrates the significant efficiency gains offered by approximate parallel approaches, especially when high accuracy is not critical. (Running the exact method with more than 6 threads will not improve the running time - see report.pdf).
    - The number of threads in our implementation affects result accuracy due to the way the problem is divided into sub-problems. Increasing threads introduces more splits, which can lead to greater error during data merging (as discussed in the report.pdf). However, this relationship is not strictly proportional; for instance, doubling threads from 4 to 8 makes the process $45\%$ faster, while accuracy only decreases by $40\%$. This demonstrates the balance we aim to achieve.


### Performance Metrics
The plots below illustrate the performance of the approximate k-NN methods (`pthread`, `openmp`, and `opencilk`) based on the results obtained from running test case 3. The evaluation was performed by varying the number of threads and examining the relationship between performance (Queries per Second) and accuracy (Neighbors Hit Rate).

### 1. Queries per Second vs. Number of Threads
![Queries per Second - Number of Threads](results/plots/queries_per_sec_threads.png)

This plot shows the query rate as the number of threads increases for the three approximate parallel k-NN methods.

**Observations:**
- **Linear Scalability**: All methods exhibit linear scaling in terms of Queries per Second as the number of threads increases. This indicates that the workload benefits significantly from parallelization.
- **Method Performance**:
  - `openmp` and `opencilk` have similar performance but for larger number of threads lag behind `pthread`.

**Key Outcome**: Increasing the number of threads significantly boosts performance, with `pthread` to outperform the other methods in absolute query rates.

---

### 2. Queries per Second vs. Neighbors Hit Rate
![Queries per Second - Neighbors Hit Rate](results/plots/queries_per_sec_neighbors_hit_rate.png)

This plot illustrates how the query rate varies with the accuracy metric (Neighbors Hit Rate) for different methods.

**Observations:**
- **Accuracy-Performance Tradeoff**: As the Neighbors Hit Rate decreases, the Queries per Second increases across all methods. This is due to less computational effort being required for approximate results with lower accuracy.
- **Method Comparison**:
  - At higher accuracy levels (lower Queries per Second), the methods are closely matched.
  - As accuracy drops, `pthread` achieves higher query rates, followed by `opencilk`, with `openmp` slightly behind.

**Key Outcome**: There is a clear tradeoff between accuracy and performance, with `pthread` offering the best performance at lower accuracy levels.

---

### Summary of Results
The table below summarizes the key trends observed from the data and plots:

| **Factor**                  | **Key Observation**                                                                                  |
|-----------------------------|-----------------------------------------------------------------------------------------------------|
| **Scalability**             | Performance scales linearly with thread count for all approximate methods, demonstrating efficient parallelization. |
| **Best Performance**        | `pthread` achieves the highest query rates when accuracy is low                                    |
| **Accuracy vs. Performance**| Higher query rates are achieved at the cost of lower Neighbors Hit Rate for all methods.            |
| **Overhead Differences**    | `openmp` and `opencilk` exhibit slightly more overhead than `pthread`, leading to lower query rates. |

These results highlight the trade-offs between performance, accuracy, and method choice when implementing parallel approximate k-NN computations.

---
