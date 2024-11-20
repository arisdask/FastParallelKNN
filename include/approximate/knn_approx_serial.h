#ifndef KNN_APPROX_SERIAL_H
#define KNN_APPROX_SERIAL_H

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../../include/utils/data_io.h"
#include "../../include/utils/distance.h"
#include "../../include/utils/mem_info.h"
#include "../../include/exact/knn_exact_serial.h"
#include "../../include/exact/knn_exact_pthread.h"

#define __ZERO__ 1e-4

/**
 * Merges two sorted arrays of distances and their corresponding indices to retain only the k smallest values.
 * 
 * @param k                 Number of nearest neighbors to keep.
 * @param existing_distances 
 *                          Pointer to the existing distances array (already sorted).
 * @param existing_indices  Pointer to the corresponding indices of `existing_distances`.
 * @param new_distances     Pointer to the new distances array to merge (already sorted).
 * @param new_indices       Pointer to the corresponding indices of `new_distances`.
 * @param final_distances   Pre-allocated array to store the merged k smallest distances.
 * @param final_indices     Pre-allocated array to store the indices corresponding to `final_distances`.
 * 
 * @return                  None (results are stored in `final_distances` and `final_indices`).
 * 
 * This function takes two pre-sorted arrays of distances (and their corresponding indices), merges them,
 * and retains only the smallest `k` distances. The merging process ensures that the resulting arrays 
 * are still sorted, maintaining their order. It is commonly used to combine results from multiple 
 * subsets of k-NN calculations.
 */
void merge_k_smallest(int k, const float* existing_distances, const int* existing_indices, 
                      const float* new_distances, const int* new_indices, 
                      float* final_distances, int* final_indices);

/**
 * Splits a dataset into three parts based on the distances of data points from a hyperplane.
 * 
 * @param dataset           Pointer to the dataset matrix (reference data points).
 * @param distances_from_hyperplane
 *                          Pre-allocated array to store the distances of each data point from the hyperplane.
 * @param dataset_length    Number of rows (data points) in the dataset.
 * @param d                 Dimensionality of each data point (number of columns in the dataset).
 * @param num_of_threads    Number of threads available for parallel computation.
 * @param accuracy          Splitting depth (controls how many subsets the data will be divided into).
 * @param _norm_            Pointer to a float variable where the norm of the hyperplane vector will be stored.
 * 
 * @return                  None (results are stored in `distances_from_hyperplane` and `_norm_`).
 * 
 * This function calculates a hyperplane that splits the dataset into three parts:
 * 1. Points that are on one side of the hyperplane.
 * 2. Points on the opposite side of the hyperplane.
 * 3. Points close to the hyperplane (based on a threshold).
 * 
 * The splitting is based on computing distances from the hyperplane for each data point. The `_norm_` 
 * value represents the norm of the hyperplane vector and is used to normalize the computed distances.
 * The function is fundamental for dividing the dataset into smaller, manageable subsets in 
 * approximate k-NN algorithms.
 */
void split_dataset(const float* dataset, float* distances_from_hyperplane, int dataset_length, int d, int num_of_threads, int accuracy, float* _norm_);



/**
 * Performs an approximate k-nearest neighbors (k-NN) search on a dataset.
 * 
 * @param dataset           Pointer to the dataset matrix (corpus == query matrix, i.e., self k-NN search).
 * @param k                 Number of nearest neighbors to find for each data point.
 * @param indices           Pre-allocated array to store indices of the k-nearest neighbors for each query 
 *                          (length `dataset_length x k`).
 * @param distances         Pre-allocated array to store the Euclidean distances to the k-nearest neighbors 
 *                          (length `dataset_length x k`).
 * @param dataset_length    Number of rows (data points) in the dataset.
 * @param d                 Dimensionality of each data point (number of columns in the dataset).
 * @param num_of_threads    Number of threads to use for parallel computation.
 * @param accuracy          Degree of approximation or tolerance level. Higher values allow for more 
 *                          subdivisions of the dataset, potentially increasing accuracy at the cost 
 *                          of additional computation time.
 * 
 * @return                  None (results are stored in the pre-allocated arrays `indices` and `distances`).
 * 
 * This function calculates approximate k-NN by dividing the dataset into subsets based on hyperplane 
 * splitting. The splitting process creates smaller chunks of the dataset that are processed individually, 
 * reducing computational complexity while maintaining reasonable accuracy. The `accuracy` parameter 
 * controls the trade-off between approximation and speed.
 */
void knn_approx_serial(const float* dataset, int k, int* indices, float* distances, int dataset_length, int d, int num_of_threads, int accuracy);


#endif // KNN_APPROX_SERIAL_H
