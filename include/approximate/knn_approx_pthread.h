#ifndef KNN_APPROX_PTHREAD_H
#define KNN_APPROX_PTHREAD_H

#include "../../include/utils/data_io.h"
#include "../../include/utils/distance.h"
#include "../../include/utils/matrix_ops.h"

void knn_approx_pthread_core(const double* corpus, const double* query, int k, int* indices, double* distances, int corpus_length, int query_length, int d);

void knn_approx_pthread(const char* data_path, int k, int* indices, double* distances, int num_of_threads, int accuracy);

#endif // KNN_APPROX_PTHREAD_H
