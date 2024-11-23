using Plots

# Choose the backend
gr()

# Data from your table, organized by method and thread count
# For simplicity, I will include only the Neighbor Hit Rates and Queries per Second for each method
# for 4, 6, 8, and 12 threads.

# Neighbor Hit Rate data for each method across threads
neighbors_hit_rate_pthread = [30.16, 21.91, 17.77, 13.49]
neighbors_hit_rate_openmp = [30.16, 21.91, 17.77, 13.49]
neighbors_hit_rate_opencilk = [30.16, 21.91, 17.77, 13.49]

# Queries per second data for each method across threads
queries_per_second_pthread = [2410.16, 3395.80, 4550.52, 7009.78]
queries_per_second_openmp = [2375.47, 3516.05, 4417.21, 5843.99]
queries_per_second_opencilk = [2395.96, 3594.56, 4244.73, 6176.14]

# Thread counts
threads = [4, 6, 8, 12]

# Plot for each method across different thread counts
p1 = plot(threads, queries_per_second_pthread, label = "pthread", lw = 2, color = :blue, marker=:circle)
plot!(p1, threads, queries_per_second_openmp, label = "openmp", lw = 2, color = :green, marker=:circle)
plot!(p1, threads, queries_per_second_opencilk, label = "opencilk", lw = 2, color = :red, marker=:circle)

# Axis labels and title for first plot
xlabel!(p1, "Number of Threads")
ylabel!(p1, "Queries per Second")
title!(p1, "Queries per Sec - Number of Threads")

# Save the first plot as a PNG file
savefig(p1, "results/plots/queries_per_sec_threads.png")

# Plot for neighbors hit rate vs queries per second
p2 = plot(neighbors_hit_rate_pthread, queries_per_second_pthread, label = "pthread", lw = 2, color = :blue, marker=:circle)
plot!(p2, neighbors_hit_rate_openmp, queries_per_second_openmp, label = "openmp", lw = 2, color = :green, marker=:circle)
plot!(p2, neighbors_hit_rate_opencilk, queries_per_second_opencilk, label = "opencilk", lw = 2, color = :red, marker=:circle)

# Axis labels and title for second plot
xlabel!(p2, "Neighbors Hit Rate")
ylabel!(p2, "Queries per Second")
title!(p2, "Queries per Sec - Neighbors Hit Rate")

# Save the second plot as a PNG file
savefig(p2, "results/plots/queries_per_sec_neighbors_hit_rate.png")
