#include <cuda.h>
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
 


// i = ithreadId
// inc = num total de threads
// Kmeans kernel

// [[x1,y1,x2,y2],[x1,y1,x2,y2]]

__global__ void kmeans_kernel(int* counts, float* sums, int interval, float* points_coords, float* clusters_coords, int num_points, int num_clusters, int dim)
{
    // Calculate point index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
 
    // Stop if index is out of bounds
    if (index >= interval)
        return;
 
    // Find the nearest cluster for current point
    int pos, p, d, i;
    float min_dist, nearest_cluster, dist, diff;
    for (p = index; p < num_points; p+=interval)
    {
        min_dist = FLT_MAX;
        nearest_cluster = 0;
        for (i = 0; i < num_clusters; i++)
        {
            // Calculate distance between current point and current cluster
            dist = 0.0;
            for (d = 0; d < dim; d++)
            {
                diff = points_coords[dim * p + d] - clusters_coords[dim * i + d];
                dist += diff * diff;
            }
     
            // Update nearest cluster if necessary
            if (dist < min_dist)
            {
                min_dist = dist;
                nearest_cluster = i;
            }
        }

        pos = index * num_clusters + nearest_cluster;
        counts[pos]++;
        for (d = 0; d < dim; d++)
            sums[dim * pos + d] += points_coords[dim * p + d];

        // Assign point to nearest cluster
        //points_clusters[index] = nearest_cluster;
    }
}
 



__global__ void update_kernel(int* clusters_counts, int* counts, float* sums, float* clusters_coords, int num_clusters, int sums_size, int dim)
{
    // Calculate thread index
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Stop if index is out of bounds
    if (index >= num_clusters)
        return;

    int d, i, c = 0, pos;
    for (d = 0; d < dim; d++)
        clusters_coords[index * dim + d] = 0;

    for (i = 0; i < sums_size; i++)
    {
        pos = i * num_clusters + index;
        c += counts[pos];
        counts[pos] = 0;
    }

    for (i = 1; i < sums_size; i++)
        for (d = 0; d < dim; d++)
        {
            pos = (i * num_clusters + index) * dim + d;
            //clusters_coords[index * dim + d] += sums[pos];
            sums[index * dim + d] += sums[pos];
            sums[pos] = 0;
        }

    for (d = 0; d < dim; d++)
    {
        pos = index * dim + d;
        clusters_coords[pos] = sums[pos] / c;
        sums[pos] = 0;
        clusters_counts[index] = c;
    }
}


void get_stats()
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int max_threads_per_block = prop.maxThreadsPerBlock;
    int max_thread_blocks = prop.maxGridSize[0];

    printf("Max threads per block: %d\n", max_threads_per_block);
    printf("Max thread blocks: %d\n", max_thread_blocks);
}

int nextPowerOfTwo(int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

void findOptimalThreadBlockSizes(int size, int &threads, int &blocks) {
  threads = (size < 512) ? nextPowerOfTwo(size) : 512;
  blocks = (size + threads - 1) / threads;
}


int run()
{
    
    int dim = 2;
    int num_points = 10000000;
    int num_clusters = 4;
    float *points_coords   = (float *) malloc(sizeof(float) * num_points * dim);
    float *clusters_coords = (float *) malloc(sizeof(float) * num_clusters * dim);
    int   *clusters_counts = (int *) malloc(sizeof(int) * num_clusters);

    int threads, blocks;

    // Initialize points and clusters
    srand(10);
    for (int i = 0; i < num_points; i++)
    {
        points_coords[dim * i] = (float) rand() / RAND_MAX;
        points_coords[dim * i + 1] = (float) rand() / RAND_MAX;
    }
    for (int i = 0; i < num_clusters; i++)
    {
        clusters_coords[dim * i] = points_coords[dim * i];
        clusters_coords[dim * i + 1] = points_coords[dim * i + 1];
    }

    // Determine interval
    int interval = sqrt(num_points * dim * 2);
    findOptimalThreadBlockSizes(interval, threads, blocks);
    
    // Timer
    clock_t start, stop;
    start = clock();

    // Set up GPU data
    int* gpu_cluster_counts;
    float* gpu_sums;
    int* gpu_counts;
    float* gpu_points_coords;
    float* gpu_clusters_coords;
    cudaMalloc(&gpu_points_coords, num_points * dim * sizeof(float));
    cudaMalloc(&gpu_clusters_coords, num_clusters * dim * sizeof(float));
    cudaMalloc(&gpu_sums, num_clusters * interval * dim * sizeof(float));
    cudaMalloc(&gpu_counts, num_clusters * interval * sizeof(int));
    cudaMalloc(&gpu_cluster_counts, num_clusters * sizeof(int));

    cudaMemset(gpu_sums, 0, num_clusters * interval * dim * sizeof(float));
    cudaMemset(gpu_counts, 0, num_clusters * interval * sizeof(int));

    // Copy data and clusters to the GPU
    cudaMemcpy(gpu_points_coords, points_coords, num_points * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_clusters_coords, clusters_coords, num_clusters * dim * sizeof(float), cudaMemcpyHostToDevice);

    // Iterate
    int limit = 21;
    while (limit > 0)
    {
        limit--;

        // Run kmeans kernel
        kmeans_kernel<<<blocks, threads>>>(gpu_counts, gpu_sums, interval, gpu_points_coords, gpu_clusters_coords, num_points, num_clusters, dim); 

        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("1 -> Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }

        // Update clusters
        update_kernel<<<1, num_clusters>>>(gpu_cluster_counts, gpu_counts, gpu_sums, gpu_clusters_coords, num_clusters, interval, dim);

        // Check for errors
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("2 -> Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }

    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(clusters_coords, gpu_clusters_coords, num_clusters * dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(clusters_counts, gpu_cluster_counts, num_clusters * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up GPU memory
    cudaFree(gpu_points_coords);
    cudaFree(gpu_clusters_coords);
    cudaFree(gpu_cluster_counts);
    cudaFree(gpu_sums);
    cudaFree(gpu_counts);

    // Timer
    stop = clock();
    double exec_time = (double)(stop - start) / CLOCKS_PER_SEC;

    // Print results
    printf("Centers:\n");
    for (int i = 0; i < num_clusters; i++)
    {
        printf("Cluster %d: ( ", i);
        for (int d = 0; d < dim; d++)
            printf("%.3f ", clusters_coords[dim * i + d]);
        printf(") : %d\n", clusters_counts[i]);
    }
    printf("Time Spent: %f seconds.\n", exec_time);

    return 0;
}


int main(int argc, char const *argv[])
{
    int r = 0;
    
    r = run();

    return r;
}