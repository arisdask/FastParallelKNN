#ifndef KNN_APPROX_PTHREAD_H
#define KNN_APPROX_PTHREAD_H

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h> // For FLT_MAX
#include "../../include/approximate/knn_approx_serial.h"
#include "../exact/knn_exact_serial.h"

#include "../../include/utils/mem_info.h"


// Thread function for processing subsets
void* knn_approx_thread(void* args);



// Main function for approximate k-NN with pthreads
void knn_approx_pthread(const float* dataset, int k, int* indices, float* distances, 
                        int dataset_length, int d, int num_of_threads, int accuracy);




// /**
//  * Core function to perform approximate k-NN search for a subset of the dataset using threads.
//  * 
//  * @param args              Pointer to the arguments structure for the thread.
//  * 
//  * @return                  None (results are stored in the arguments structure passed via `args`).
//  * 
//  * This function is executed by each thread and operates on a subset of the dataset. It calculates 
//  * the approximate k-NN for the assigned subset using the `knn_approx_serial` function and updates 
//  * the shared results in the arguments structure.
//  */
// void* knn_approx_pthread_core(void* args);


// /**
//  * Performs approximate k-NN search on the entire dataset using pthreads for parallel computation.
//  * 
//  * @param dataset           Pointer to the dataset matrix (reference data points).
//  * @param k                 Number of nearest neighbors to find.
//  * @param indices           Pre-allocated array to store indices of the k-nearest neighbors for each query 
//  *                          (length `dataset_length x k`).
//  * @param distances         Pre-allocated array to store the Euclidean distances to the k-nearest neighbors 
//  *                          (length `dataset_length x k`).
//  * @param dataset_length    Number of rows (data points) in the dataset.
//  * @param d                 Dimensionality of each data point (number of columns in the dataset).
//  * @param num_of_threads    Number of threads to use for parallel computation.
//  * @param accuracy          Accuracy level (determines the number of splits and subset sizes).
//  * 
//  * @return                  None (results are stored in the pre-allocated arrays `indices` and `distances`).
//  * 
//  * This function partitions the dataset into smaller subsets based on the accuracy level, assigns 
//  * each subset to a thread, and computes the k-NN for each subset in parallel using the `knn_approx_serial` 
//  * function. The results from all subsets are combined to produce the final k-NN results.
//  */
// void knn_approx_pthread(const float* dataset, int k, int* indices, float* distances,
//                         int dataset_length, int d, int num_of_threads, int accuracy);

#endif // KNN_APPROX_PTHREAD_H
