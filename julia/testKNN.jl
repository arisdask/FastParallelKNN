include("knnAlgorithms/knnExactSerial.jl")

global ZERO = 0.01

############################  knn_exact_serial results test ############################
# ==================================================================================== #
# C: Matrix of size (n_samples, d) where each row is a data point
# Q: Matrix of size (m_samples, d) where each row is a query point
# k: Number of nearest neighbors to find
C = copy(load_hdf5(joinpath(@__DIR__, "../data/sift-128-euclidean.hdf5"), "train")')
Q = copy(load_hdf5(joinpath(@__DIR__, "../data/sift-128-euclidean.hdf5"), "test")')
k = 5

global idx = NaN
global dst = NaN

@time idx, dst = knn_exact_serial(C, Q, k, +1)
C = nothing
Q = nothing
GC.gc()

neighbors = copy(load_hdf5(joinpath(@__DIR__, "../data/sift-128-euclidean.hdf5"), "neighbors")')
distances = copy(load_hdf5(joinpath(@__DIR__, "../data/sift-128-euclidean.hdf5"), "distances")')

neighbor_errors = 0
distance_errors = 0

for i in 1:1:size(neighbors, 1)
    for j in 1:1:k
        # It's minus 1 ( - 1 ) because the data inside the dataset start at zero-index
        if abs(idx[i, j] - neighbors[i, j] - 1) > ZERO
            neighbor_errors += 1
        end
    end
end
println("Neighbors Mismatch Percentage: $(100 - neighbor_errors * 100 / (k * size(neighbors, 1)))%")
println(" ")

for i in 1:1:size(neighbors, 1)
    for j in 1:1:k
        if abs(sqrt(dst[i, j]) - distances[i, j]) > ZERO
            distance_errors += 1
        end
    end
end
println("Distances Mismatch Percentage: $(100 - distance_errors * 100 / (k * size(neighbors, 1)))%")

neighbors = nothing
distances = nothing