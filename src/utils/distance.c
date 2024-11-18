#include "../../include/utils/distance.h"

void distance_square_matrix(const float* corpus, const float* query, float* D, int corpus_length, int query_length, int d) {
    // Step 1: Compute -2 * (Q * C^T)
    // Use OpenBLAS to perform the matrix multiplication
    // D = -2 * Q * C^T
    // Q is of size (query_length x d) and C^T is of size (d x corpus_length)
    // Resulting matrix D will be of size (query_length x corpus_length)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                query_length, corpus_length, d,
                -2.0f, query, d, corpus, d,
                0.0f, D, corpus_length);
    
    // Step 2: Compute squared norms for the query rows
    float* query_norms = (float*)malloc(query_length * sizeof(float));
    for (int i = 0; i < query_length; i++) {
        query_norms[i] = 0.0f;
        for (int k = 0; k < d; k++) {
            query_norms[i] += query[i * d + k] * query[i * d + k];
        }
    }

    // Step 3: Compute squared norms for the corpus rows
    float* corpus_norms = (float*)malloc(corpus_length * sizeof(float));
    for (int i = 0; i < corpus_length; i++) {
        corpus_norms[i] = 0.0f;
        for (int k = 0; k < d; k++) {
            corpus_norms[i] += corpus[i * d + k] * corpus[i * d + k];
        }
    }

    // Step 4: Add the norms to the result matrix D
    // D[i, j] correspond to the distance between the j-th corpus sample and i-th query sample
    for (int i = 0; i < query_length; i++) {
        for (int j = 0; j < corpus_length; j++) {
            D[i * corpus_length + j] += query_norms[i] + corpus_norms[j];
        }
    }

    free(corpus_norms);
    free(query_norms);
}
