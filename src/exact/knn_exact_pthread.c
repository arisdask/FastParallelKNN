#include "../../include/exact/knn_exact_pthread.h"

void* knn_exact_pthread_core(void* args) {
    knn_thread_args_t* thread_args = (knn_thread_args_t*)args;

    // Calculate start index for the query chunk, handled by this thread
    int chunk_size = (thread_args->query_length + thread_args->num_of_threads - 1) / thread_args->num_of_threads;  // Divide queries evenly
    int q_start = thread_args->thread_id * chunk_size;
    int q_chunk_length = (q_start + chunk_size < thread_args->query_length) ? chunk_size : (thread_args->query_length - q_start);

    // Allocate and load the query chunk
    const float* query_chunk = &thread_args->query[q_start * thread_args->d];

    // Perform k-NN search on the assigned query chunk
    knn_exact_serial(
        thread_args->corpus,
        query_chunk,
        thread_args->k,
        &thread_args->indices[q_start * thread_args->k],
        &thread_args->distances[q_start * thread_args->k],
        thread_args->corpus_length,
        q_chunk_length,
        thread_args->d,
        thread_args->num_of_threads
    );

    pthread_exit(NULL);
}


void knn_exact_pthread(const float* corpus, const float* query, int k, int* indices, float* distances, int corpus_length, int query_length, int d, int num_of_threads) {
    // Check if there are more threads than queries
    if (num_of_threads > query_length) { num_of_threads = query_length; }

    // Array of thread handles
    pthread_t* threads = (pthread_t*)malloc(num_of_threads * sizeof(pthread_t));
    knn_thread_args_t* thread_args = (knn_thread_args_t*)malloc(num_of_threads * sizeof(knn_thread_args_t));

    // Initialize and create threads
    for (int i = 0; i < num_of_threads; ++i) {
        // Set up arguments for each thread
        thread_args[i].corpus           =   corpus;
        thread_args[i].query            =   query;
        thread_args[i].k                =   k;
        thread_args[i].indices          =   indices;
        thread_args[i].distances        =   distances;
        thread_args[i].corpus_length    =   corpus_length;
        thread_args[i].query_length     =   query_length;
        thread_args[i].d                =   d;
        thread_args[i].thread_id        =   i;
        thread_args[i].num_of_threads   =   num_of_threads;

        // Create the thread
        if (pthread_create(&threads[i], NULL, knn_exact_pthread_core, &thread_args[i]) != 0) {
            fprintf(stderr, "knn_exact_pthread: Error creating thread %d\n", i);
            free(threads);
            free(thread_args);
            return;
        }
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_of_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    // Cleanup
    free(threads);
    free(thread_args);
}
