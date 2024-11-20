#!/bin/bash

# This script runs the `knn_project` executable with the specified parameters provided as command-line arguments.
# Usage examples:
#   ./run_knn.sh [method] [num_of_threads] [data_path] [corpus_name] [query_name] [k] [compare_results (optional)] [neighbors (optional)] [distances (optional)]
#       -> Add *all* the optional values to compare the k-NN's results with your own results:
#   ./run_knn.sh 4 3 data/sift-128-euclidean.hdf5 train test 100 data/sift-128-euclidean.hdf5 neighbors distances
#       -> Add *none* of the optional values to just run the code and store the output matrices in the results/data_knn:
#   ./run_knn.sh 2 5 data/sift-128-euclidean.hdf5 test test 5

#  0 - Run All for specific num_of_threads and compare the results with the `knn_exact_serial` results (which can be manualy tested)
#  1 - knn_exact_serial
#  2 - knn_exact_pthread
#  3 - knn_exact_openmp
#  4 - knn_exact_opencilk
#  5 - knn_approx_serial
#  6 - knn_approx_pthread
#  7 - knn_approx_openmp
#  8 - knn_approx_opencilk
#  9 - Run Approx Tests

# Check the number of arguments
if [ "$#" -lt 6 ]; then
    echo "Usage: $0 [method] [num_of_threads] [data_path] [corpus_name] [query_name] [k] [compare_results (optional)] [neighbors (optional)] [distances (optional)]"
    exit 1
fi

# Parse command line arguments
METHOD=$1
NUM_THREADS=$2
DATA_PATH=$3
CORPUS_NAME=$4
QUERY_NAME=$5
K_VALUE=$6
COMPARE_RESULTS=${7:-""}
NEIGHBORS=${8:-""}
DISTANCES=${9:-""}      # Optional argument for distances dataset

# Validate the method input
if ! [[ "$METHOD" =~ ^[0-9]$ ]] || [ "$METHOD" -lt 0 ] || [ "$METHOD" -gt 9 ]; then
    echo "Error: Method should be a number from 0 to 9."
    exit 1
fi

# Determine which executable to run
if [[ "$METHOD" -eq 4 || "$METHOD" -eq 8 ]]; then
    # Build project with Clang
    echo "Building Project with Makefile.clang..."
    make -f Makefile.clang clean
    make -f Makefile.clang
    echo "Building with Makefile.clang: Done"

    # For OpenCilk methods, use the `knn_project_clang` executable
    export CILK_NWORKERS=$NUM_THREADS
    EXECUTABLE="./knn_project_clang"
else
    echo "Building Project with Makefile.gcc..."
    make -f Makefile.gcc clean
    make -f Makefile.gcc
    echo "Building with Makefile.gcc: Done"

    # Use the default `knn_project` executable
    EXECUTABLE="./knn_project"
fi

COMMAND="$EXECUTABLE $METHOD $NUM_THREADS $DATA_PATH $CORPUS_NAME $QUERY_NAME $K_VALUE $COMPARE_RESULTS $NEIGHBORS $DISTANCES"

# Display the constructed command
echo "Running command: $COMMAND"

# Execute the command
$COMMAND

# Check again if method is 0 or 9 to run the knn_exact_opencilk or knn_approx_opencilk
if [[ "$METHOD" -eq 0 || "$METHOD" -eq 9 ]]; then
    echo "Building Project with Makefile.clang..."
    make -f Makefile.clang clean
    make -f Makefile.clang
    echo "Building with Makefile.clang: Done"

    export CILK_NWORKERS=$NUM_THREADS
    EXECUTABLE="./knn_project_clang"

    # For OpenCilk methods, use the `knn_project_clang` executable
    COMMAND="$EXECUTABLE $METHOD $NUM_THREADS $DATA_PATH $CORPUS_NAME $QUERY_NAME $K_VALUE"

    # Display the constructed command
    echo "Running command: $COMMAND"

    # Execute the command
    $COMMAND
fi
