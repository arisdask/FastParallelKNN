#ifndef KNN_EXACT_PTHREAD_H
#define KNN_EXACT_PTHREAD_H

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../include/exact/knn_exact_serial.h"

// Structure to hold arguments for each thread
typedef struct {
    const float*    corpus;
    const float*    query;
    int             k;
    int*            indices;
    float*          distances;
    int             corpus_length;
    int             query_length;
    int             d;
    int             thread_id;
    int             num_of_threads;
} knn_thread_args_t;

void partial_sort(float* distances, int* indices, int length, int k);

// Thread function to perform k-NN search on a subset of queries
void* knn_exact_pthread_core(void* args);


/**
 * Wrapper function to perform k-nearest neighbor search using a multi-threaded approach with Pthreads,
 * handling memory constraints by processing in blocks using multiple threads.
 * 
 * @param corpus            Pointer to the corpus matrix (reference data points)
 * @param query             Pointer to the query matrix (data points to compare)
 * @param k                 Number of nearest neighbors to find
 * @param indices           Pre-allocated array to store indices of the k-nearest neighbors for each query (length `query_length x k`)
 * @param distances         Pre-allocated array to store the Euclidean distances to the k-nearest neighbors (length `query_length x k`)
 * @param corpus_length     Number of rows (data points) in the corpus
 * @param query_length      Number of rows (data points) in the query
 * @param d                 Dimensionality of each data point (number of columns in corpus/query)
 * @param num_of_threads    Number of threads to use for parallel processing
 * 
 * @return                  None (results are stored in the pre-allocated arrays `indices` and `distances`)
 */
void knn_exact_pthread(const float* corpus, const float* query, int k, int* indices, float* distances, int corpus_length, int query_length, int d, int num_of_threads);

#endif // KNN_EXACT_PTHREAD_H
