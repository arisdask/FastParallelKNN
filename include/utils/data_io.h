#ifndef DATA_IO_H
#define DATA_IO_H

#include <stdlib.h>
#include <stdio.h>
#include <hdf5/serial/hdf5.h>  //sudo apt-get install libhdf5-dev

/**
 * Loads data from an HDF5 file.
 * 
 * @param filename      Path to the HDF5 file.
 * @param dataset_name  Name of the dataset within the HDF5 file to load.
 * @param n             Pointer to store the number of rows loaded.
 * @param d             Pointer to store the dimensionality (number of columns) of each row.
 * 
 * @return              Pointer to a dynamically allocated array containing the loaded data,
 *                      or NULL if an error occurs. The caller is responsible for freeing the memory.
 */
float* load_hdf5(const char* filename, const char* dataset_name, int* n, int* d);

/**
 * Save data to an HDF5 file.
 * 
 * @param filename      Path to the HDF5 file to create or overwrite.
 * @param dataset_name  Name of the dataset within the HDF5 file to save.
 * @param data          Pointer to the data to save, organized as a 1D-array.
 * @param n             Number of rows in the dataset.
 * @param d             Dimensionality (number of columns) of each row in the dataset.
 * 
 * @return              0 on success, -1 on failure.
 */
int save_hdf5(const char* filename, const char* dataset_name, const float* data, int n, int d);

#endif // DATA_IO_H
