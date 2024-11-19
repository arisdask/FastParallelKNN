#!/bin/bash

read -p "Enter number of threads: " num

echo "Building Project with Makefile.gcc..."
make -f Makefile.gcc clean
make -f Makefile.gcc
echo "Building with Makefile.gcc: Done"

echo "Running Project, $num threads..."
time ./knn_project $num > results/logs/main.log
echo "Running: Done"

echo " "

echo "Building Project with Makefile.clang..."
make -f Makefile.clang clean
make -f Makefile.clang
echo "Building with Makefile.clang: Done"

export CILK_NWORKERS=$num

echo "Running OpenCilk k-NN, $num threads..."
time ./knn_project_clang $num > results/logs/main_opencilk.log
echo "Running: Done"
