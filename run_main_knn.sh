#!/bin/bash

read -p "Enter number of threads: " num

echo "Building Project with Makefile.gcc..."
make -f Makefile.gcc clean
make -f Makefile.gcc
echo "Building with Makefile.gcc: Done"

echo "Running Project, $num threads..."
time ./knn_project $num > results/logs/main.log
echo "Running: Done"