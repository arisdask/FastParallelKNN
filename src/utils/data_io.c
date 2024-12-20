#include "../../include/utils/data_io.h"

float* load_hdf5(const char* filename, const char* dataset_name, int* n, int* d) {
    hid_t file_id, dataset_id, space_id;
    hsize_t dims[2];
    float* data = NULL;

    // Open the HDF5 file:
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "load_hdf5: Error opening HDF5 file: %s\n", filename);
        return NULL;
    }

    // Open the HDF5 dataset:
    dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "load_hdf5: Error opening dataset: %s in file: %s\n", dataset_name, filename);
        H5Fclose(file_id);
        return NULL;
    }
    
    // Get dataset dimensions:
    space_id = H5Dget_space(dataset_id);
    H5Sget_simple_extent_dims(space_id, dims, NULL);
    *n = (int)dims[0];
    *d = (int)dims[1];

    // Allocate memory and read data:
    data = (float*)malloc((*n) * (*d) * sizeof(float));
    if (H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0) {
        fprintf(stderr, "load_hdf5: Error reading data: %s in file: %s\n", dataset_name, filename);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    // Close resources
    H5Sclose(space_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    return data;
}


int save_float_hdf5(const char* filename, const char* dataset_name, const float* data, int n, int d) {
    hid_t file_id;
    hid_t dataset_id;
    hid_t dataspace_id;
    hsize_t dims[2] = {n, d};

    // Check if the file already exists
    struct stat buffer;
    if (stat(filename, &buffer) == 0) {
        // File exists, open it
        file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
        if (file_id < 0) {
            fprintf(stderr, "save_hdf5: Error opening existing file: %s\n", filename);
            return -1;
        }
    } else {
        // File does not exist, create it
        file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file_id < 0) {
            fprintf(stderr, "save_hdf5: Error creating file: %s\n", filename);
            return -1;
        }
    }

    // Check if the dataset already exists
    if (H5Lexists(file_id, dataset_name, H5P_DEFAULT) > 0) {
        // Delete the dataset by name
        if (H5Ldelete(file_id, dataset_name, H5P_DEFAULT) < 0) {
            fprintf(stderr, "save_hdf5: Error deleting old dataset: %s in file: %s\n", dataset_name, filename);
            H5Fclose(file_id);
            return -1;
        }
    }

    // Create the data space for the new dataset
    dataspace_id = H5Screate_simple(2, dims, NULL);
    if (dataspace_id < 0) {
        fprintf(stderr, "save_hdf5: Error creating dataspace\n");
        H5Fclose(file_id);
        return -1;
    }

    // Create the dataset with default properties
    dataset_id = H5Dcreate2(file_id, dataset_name, H5T_NATIVE_FLOAT, dataspace_id,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "save_hdf5: Error creating dataset: %s\n", dataset_name);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return -1;
    }

    // Write the data to the dataset
    if (H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0) {
        fprintf(stderr, "save_hdf5: Error writing data to dataset: %s\n", dataset_name);
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return -1;
    }

    // Close the dataset and file
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);

    return 0; // Success
}


int save_int_hdf5(const char* filename, const char* dataset_name, const int* data, int n, int d) {
    hid_t file_id;
    hid_t dataset_id;
    hid_t dataspace_id;
    hsize_t dims[2] = {n, d};

    // Check if the file already exists
    struct stat buffer;
    if (stat(filename, &buffer) == 0) {
        // File exists, open it
        file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
        if (file_id < 0) {
            fprintf(stderr, "save_hdf5: Error opening existing file: %s\n", filename);
            return -1;
        }
    } else {
        // File does not exist, create it
        file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file_id < 0) {
            fprintf(stderr, "save_hdf5: Error creating file: %s\n", filename);
            return -1;
        }
    }

    // Check if the dataset already exists
    if (H5Lexists(file_id, dataset_name, H5P_DEFAULT) > 0) {
        // Delete the dataset by name
        if (H5Ldelete(file_id, dataset_name, H5P_DEFAULT) < 0) {
            fprintf(stderr, "save_hdf5: Error deleting old dataset: %s in file: %s\n", dataset_name, filename);
            H5Fclose(file_id);
            return -1;
        }
    }

    // Create the data space for the new dataset
    dataspace_id = H5Screate_simple(2, dims, NULL);
    if (dataspace_id < 0) {
        fprintf(stderr, "save_hdf5: Error creating dataspace\n");
        H5Fclose(file_id);
        return -1;
    }

    // Create the dataset with default properties for integer data
    dataset_id = H5Dcreate2(file_id, dataset_name, H5T_NATIVE_INT, dataspace_id,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "save_hdf5: Error creating dataset: %s\n", dataset_name);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return -1;
    }

    // Write the data to the dataset
    if (H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0) {
        fprintf(stderr, "save_hdf5: Error writing data to dataset: %s\n", dataset_name);
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return -1;
    }

    // Close the dataset and file
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);

    return 0; // Success
}

