#ifndef KNN_APPROX_OPENCILK_H
#define KNN_APPROX_OPENCILK_H

#include "../../include/utils/data_io.h"
#include "../../include/utils/distance.h"
#include "../../include/utils/matrix_ops.h"

void knn_approx_opencilk_core(const double* corpus, const double* query, int k, int* indices, double* distances, int corpus_length, int query_length, int d);

void knn_approx_opencilk(const char* data_path, int k, int* indices, double* distances, int num_of_threads, int accuracy);

#endif // KNN_APPROX_OPENCILK_H
