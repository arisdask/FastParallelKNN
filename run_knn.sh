#!/bin/bash

# This script runs the `knn_project` executable with the specified parameters provided as command-line arguments.
# Usage examples:
#   ./run_knn.sh [method] [num_of_threads] [data_path] [corpus_name] [query_name] [k] [compare_results (optional)] [neighbors (optional)] [distances (optional)]
#       -> Add *all* the optional values to compare the k-NN's results with the expected (for method 0 only!)
#       -> Add *none* of the optional values when you use auto-generated random dataset (methods 1 and 2)

# Ex: ./run_knn.sh 0 4 data/sift-128-euclidean.hdf5 train test 100 data/sift-128-euclidean.hdf5 neighbors distances
# Ex: ./run_knn.sh 1 4 null null null 100
# Ex: ./run_knn.sh 2 5 null null null 100

#  0 - Run all the *exact* knn functions and evaluate/compare the results based on the given dataset
#  1 - Run all the *approx* knn functions and evaluate/compare the results (based on the exact results of an exact knn)
#      Keep in mind that the approximate solutions solve only the all-to-all k-NN problem in which C == Q
#  2 - Random Data Test for knn_approx_pthread (Playground)
#  3 - You can add your own custom tests here!

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


echo "Building Project with Makefile.gcc..."
make -f Makefile.gcc clean
make -f Makefile.gcc
echo "Building with Makefile.gcc: Done"

# Use the default `knn_project` executable
EXECUTABLE="./knn_project"
COMMAND="$EXECUTABLE $METHOD $NUM_THREADS $DATA_PATH $CORPUS_NAME $QUERY_NAME $K_VALUE $COMPARE_RESULTS $NEIGHBORS $DISTANCES"

echo " "
# Display the constructed command
echo "Running command: $COMMAND"
echo " "

# Execute the command
$COMMAND
echo " "

# Determine which executable to run
if [[ "$METHOD" -eq 0 || "$METHOD" -eq 1 ]]; then
    # Build project with Clang
    echo "Building Project with Makefile.clang..."
    make -f Makefile.clang clean
    make -f Makefile.clang
    echo "Building with Makefile.clang: Done"

    # For OpenCilk methods, use the `knn_project_clang` executable
    export CILK_NWORKERS=$NUM_THREADS
    EXECUTABLE="./knn_project_clang"
    COMMAND="$EXECUTABLE $METHOD $NUM_THREADS $DATA_PATH $CORPUS_NAME $QUERY_NAME $K_VALUE $COMPARE_RESULTS $NEIGHBORS $DISTANCES"

    echo " "
    # Display the constructed command
    echo "Running command: $COMMAND"
    echo " "

    # Execute the command
    $COMMAND
    echo " "
fi
