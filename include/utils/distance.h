#ifndef DISTANCE_H
#define DISTANCE_H

#include <cblas.h>    //sudo apt-get install libopenblas-dev
#include <math.h>
#include <stdlib.h>

/**
 * Computes the squared Euclidean distances between each pair of rows from two matrices (`corpus` and `query`) 
 * and stores them in a distance matrix `D`.
 * 
 * @param corpus        Pointer to the corpus matrix (each row represents a data point)
 * @param query         Pointer to the query matrix (each row represents a data point to be compared against the corpus)
 * @param D             Pre-allocated matrix to store the squared Euclidean distances, size `query_length x corpus_length`
 * @param corpus_length Number of rows (data points) in the `corpus`
 * @param query_length  Number of rows (data points) in the `query`
 * @param d             Dimensionality of each data point (number of columns in `corpus` and `query`)
 * 
 * @return              None (results are stored in the pre-allocated matrix D)
 */
void distance_square_matrix(const float* corpus, const float* query, float* D, int corpus_length, int query_length, int d);

#endif // DISTANCE_H
