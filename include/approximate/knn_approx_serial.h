#ifndef KNN_APPROX_SERIAL_H
#define KNN_APPROX_SERIAL_H

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "../../include/utils/data_io.h"
#include "../../include/utils/distance.h"
#include "../../include/utils/mem_info.h"
#include "../../include/exact/knn_exact_serial.h"

#define __ZERO__ 1e-4

void split_dataset(const float* dataset, float* distances_from_hyperplane, int dataset_length, int d, int num_of_threads, int accuracy, float* _norm_);

void knn_approx_serial_core(const float* dataset, int k, int* indices, float* distances, int dataset_length, int d, int num_of_threads, int accuracy);


/**
 * @param dataset           Pointer to the corpus == query matrix (reference data points)
 * @param k                 Number of nearest neighbors to find
 * @param indices           Pre-allocated array to store indices of the k-nearest neighbors for each query (length `query_length x k`)
 * @param distances         Pre-allocated array to store the Euclidean distances to the k-nearest neighbors (length `query_length x k`)
 * @param dataset_length     Number of rows (data points) in the corpus
 * @param d                 Dimensionality of each data point (number of columns in corpus/query)
 * @param num_of_threads    Number of threads which run this function simultaneously.
 * @param accuracy
 * 
 * @return                  None (results are stored in the pre-allocated arrays `indices` and `distances`)
 */
void knn_approx_serial(const float* dataset, int k, int* indices, float* distances, int dataset_length, int d, int num_of_threads, int accuracy);

#endif // KNN_APPROX_SERIAL_H
