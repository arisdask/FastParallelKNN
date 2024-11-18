using StatsBase      # import Pkg; Pkg.add("StatsBase")
using LinearAlgebra  # import Pkg; Pkg.add("LinearAlgebra")
using HDF5           # import Pkg; Pkg.add("HDF5")


# Estimate the amount of usable memory available on the system (Linux specific).  
# Returns:
#  - size of usable memory in bytes.
function get_usable_memory()
    # This should be replaced with a proper memory check for your environment
    # In Linux, use `Sys.free_memory()` or similar system utilities
    return Sys.free_memory() * 0.7  # Change the scalar to determine the usable memory!
end


# Loads data from an HDF5 file.
# `filename`      Path to the HDF5 file.
# `dataset_name`  Name of the dataset within the HDF5 file to load.
# Returns:
#  - Matrix containing the loaded data.
function load_hdf5(filename::String, dataset_name::String)
    file = h5open(filename, "r")     # Explicitly open the file
    data = read(file[dataset_name])  # Read the dataset
    close(file)                      # Explicitly close the file
    return data
end


# Core function to compute the k-nearest neighbors (k-NN) between two matrices using a serial approach.
# `corpus`: Matrix where each row is a data point from the reference set.
# `query`:  Matrix where each row is a data point from the query set.
# `k`:      The number of nearest neighbors to identify for each query point.
# Returns: 
#  - `indices`   of the k-nearest neighbors for each query (length `query_length x k`)
#  - `distances` to the k-nearest neighbors (*squared*) (length `query_length x k`)
function knn_exact_serial_core(corpus, query, k)
    query_length, _ = size(query)       # Number of rows (samples) in the query

    # Preallocate the result matrices
    indices = Matrix{Int}(undef, query_length, k)
    distances = Matrix{Float32}(undef, query_length, k)

    # Step 1: Compute squared norms for each row in the corpus and query
    corpus_norms = sum(corpus.^2, dims=2)  # Sum of squares for each corpus row
    query_norms = sum(query.^2, dims=2)    # Sum of squares for each query row

    # Step 2: Compute the distance matrix
    dist_matrix = -2 * (query * corpus')
    dist_matrix .+= corpus_norms' .+ query_norms # corpus_norms and query_norms are column vectors


    # Step 3: Find the k nearest neighbors using "partialsortperm"
    for i in 1:query_length
        # Extract the distances for the i-th query
        dists = dist_matrix[i, :]

        # Get the k smallest distances and their indices
        sorted_indices = partialsortperm(dists, 1:k)
        sorted_dists = dists[sorted_indices]
        
        # Store the indices and distances
        indices[i, :] .= sorted_indices
        distances[i, :] .= sorted_dists
    end

    return indices, distances
end


# Parameters:
# - `corpus`        A matrix of data points that form the reference dataset.
# - `query`         A matrix of data points that need neighbors identified.
# - `k`             The number of nearest neighbors to find for each query point.
# - `num_of_blocks` Number of blocks to split the query for processing.
function knn_exact_serial(corpus, query, k, num_of_blocks)
    println("knn_exact_serial: Start")
    corpus_length, d = size(corpus)
    query_length = size(query, 1)

    # Preallocate result arrays for the entire dataset
    final_indices = Matrix{Int}(undef, query_length, k)
    final_distances = Matrix{Float32}(undef, query_length, k)

    q_start = 1
    while q_start <= query_length
        # Calculate max chunk length
        max_chunk_length = (num_of_blocks <= 0) ? 
            div((get_usable_memory() - d * (corpus_length + 1) * sizeof(Float64) - 2 * k * sizeof(Float64)),
                ((corpus_length + d + 2 * k) * sizeof(Float64))) :
            div(query_length, num_of_blocks)

        if max_chunk_length == 0
            max_chunk_length = 1
        end
        max_chunk_length = Int(max_chunk_length)
        # println(max_chunk_length)
        
        # Adjust chunk boundaries
        q_chunk_length = min(max_chunk_length, query_length - q_start + 1)

        # Perform k-NN search for the current chunk using core function
        chunk_indices, chunk_distances = knn_exact_serial_core(corpus, query[q_start:q_start + q_chunk_length - 1, :], k)

        # Fill the global indices and distances matrices
        final_indices[q_start:q_start + q_chunk_length - 1, :] .= chunk_indices
        final_distances[q_start:q_start + q_chunk_length - 1, :] .= chunk_distances

        # Move to the next chunk/block
        q_start += max_chunk_length
        println("knn_exact_serial: $(100 * q_start / query_length)% Complete...")
        GC.gc()
    end
    
    println("knn_exact_serial: Done")
    # Squared Distances!
    return final_indices, final_distances
end


############################  knn_exact_serial results test ############################
# ==================================================================================== #
for i in 1:1:10
    # corpus: Matrix of size (n_samples, d) where each row is a data point
    # query: Matrix of size (m_samples, d) where each row is a query point
    # k: Number of nearest neighbors to find
    C = copy(load_hdf5(joinpath(@__DIR__, "../../data/sift-128-euclidean.hdf5"), "train")')
    Q = copy(load_hdf5(joinpath(@__DIR__, "../../data/sift-128-euclidean.hdf5"), "test")')
    k = 5

    global idx = NaN
    global dst = NaN

    @time idx, dst = knn_exact_serial(C, Q, k, -1)
    C = nothing
    Q = nothing
    GC.gc()

    neighbors = copy(load_hdf5(joinpath(@__DIR__, "../../data/sift-128-euclidean.hdf5"), "neighbors")')
    distances = copy(load_hdf5(joinpath(@__DIR__, "../../data/sift-128-euclidean.hdf5"), "distances")')

    err_cnt = 0
    for i in 1:1:size(neighbors, 1)
        for j in 1:1:k
            # It's minus 1 ( - 1 ) because the data start at zero in the dataset.
            if abs(idx[i, j] - neighbors[i, j] - 1) > 0.01
                # println("Error at neighbors: $i, $j, idx: $(idx[i, j] - 1), neighbors: $(neighbors[i, j])")
                err_cnt += 1
            end
        end
    end
    println("Neighbors Accuracy: $(100 - err_cnt * 100 / (k * size(neighbors, 1)))%")
    println(" ")

    err_cnt = 0
    for i in 1:1:size(neighbors, 1)
        for j in 1:1:k
            if abs(sqrt(dst[i, j]) - distances[i, j]) > 0.01
                # println("Error at distances: $i, $j, dst: $(dst[i, j]), distances: $(distances[i, j])")
                err_cnt += 1
            end
        end
    end
    println("Distances Accuracy: $(100 - err_cnt * 100 / (k * size(neighbors, 1)))%")
    println("=============================================================================")
end