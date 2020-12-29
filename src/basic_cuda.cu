//
// Created by petioptrv on 2020-10-10.
//

#include <stdio.h>

#include "basic_cuda.h"
#include "helpers.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ double atomicAddc(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void assign_points_to_clusters(options_t *opts, int *n_vals,
        double *coords_d_points, int *cluster_d_points, double *min_dist_d_points,
        double *coord_d_centroids, double *prev_coord_d_centroids,
        double *n_points_d_centroids, double *sums_d_centroids) {
    int p = threadIdx.x + blockIdx.x * blockDim.x;
    int ops_count = opts->num_cluster * opts->dims;

    if (p < ops_count) {
        if (p < opts->num_cluster) {
            n_points_d_centroids[p] = 0;
        }
        sums_d_centroids[p] = 0;
        prev_coord_d_centroids[p] = coord_d_centroids[p];
    }

    if (p >= *n_vals) {
        return;
    }

    __syncthreads();

    min_dist_d_points[p] = __DBL_MAX__;

    long point_coord_offset = p * opts->dims;

    int cluster;
    for (int c = 0; c != opts->num_cluster; ++c) {
        double dist = 0;
        long centroid_coord_offset = c * opts->dims;
        for (int d = 0; d != opts->dims; d++) {

            // todo: unroll this further

            dist += (coord_d_centroids[centroid_coord_offset + d] - coords_d_points[point_coord_offset + d])
                    * (coord_d_centroids[centroid_coord_offset + d] - coords_d_points[point_coord_offset + d]);
        }

        if (dist < min_dist_d_points[p]) {
            min_dist_d_points[p] = dist;
            cluster_d_points[p] = c;
            cluster = c;
        }
    }

    atomicAddc(&n_points_d_centroids[cluster], 1);
    long centroid_coord_offset = cluster * opts->dims;
    for (int d = 0; d != opts->dims; ++d) {
        atomicAddc(&sums_d_centroids[centroid_coord_offset + d], coords_d_points[point_coord_offset + d]);
    }
}

__global__ void compute_new_centroids(options_t *opts,
        double *coord_d_centroids, double *prev_coord_d_centroids,
        double *n_points_d_centroids, double *sums_d_centroids, bool *converged_d_centroids) {
    int p = threadIdx.x + blockIdx.x * blockDim.x;
    int ops_count = opts->num_cluster * opts->dims;

    if (p >= ops_count) {
        return;
    }

    int c = p / opts->dims;

    coord_d_centroids[p] = sums_d_centroids[p] / n_points_d_centroids[c];

    __syncthreads();

    if (p % opts->dims == 0) {
        double dist = 0;
        for (int d = 0; d != opts->dims; d++) {

            // todo: unroll this further

            dist += (coord_d_centroids[p + d] - prev_coord_d_centroids[p + d])
                    * (coord_d_centroids[p + d] - prev_coord_d_centroids[p + d]);
        }
        dist = sqrt(dist);
        if (dist > opts->threshold) {
            converged_d_centroids[c] = false;
        } else {
            converged_d_centroids[c] = true;
        }
    }
}

void k_means_cuda_basic(point_t *points, centroid_t *centroids, options_t *opts, int *n_vals, timing_t *timing) {
    double *coords_d_points;
    int *cluster_d_points;
    double *min_dist_d_points;

    double *coord_d_centroids;
    double *prev_coord_d_centroids;
    double *n_points_d_centroids;
    double *sums_d_centroids;
    bool *converged_d_centroids;

    options_t *d_opts;
    int *d_n_vals;

    init_device_mem(points, centroids, opts, n_vals,
            &coords_d_points, &cluster_d_points, &min_dist_d_points,
            &coord_d_centroids, &prev_coord_d_centroids,
            &n_points_d_centroids, &sums_d_centroids, &converged_d_centroids,
            &d_opts, &d_n_vals);

    gpuErrchk( cudaPeekAtLastError() )
    gpuErrchk( cudaDeviceSynchronize() )

    int threads = opts->thread_count;
    int blocks;
    bool converged = false;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    int e;
    for (e = 0; e != opts->max_num_iter or converged; ++e) {
        blocks = (*n_vals + threads - 1) / threads;
        assign_points_to_clusters<<<blocks, threads>>>(d_opts, d_n_vals,
                coords_d_points, cluster_d_points, min_dist_d_points,
                coord_d_centroids, prev_coord_d_centroids, n_points_d_centroids, sums_d_centroids);
        gpuErrchk( cudaPeekAtLastError() )
        gpuErrchk( cudaDeviceSynchronize() )

        blocks = (opts->num_cluster * opts->dims + threads - 1) / threads;
        compute_new_centroids<<<blocks, threads>>>(d_opts,
                coord_d_centroids, prev_coord_d_centroids, n_points_d_centroids, sums_d_centroids,
                converged_d_centroids);
        gpuErrchk( cudaPeekAtLastError() )
        gpuErrchk( cudaDeviceSynchronize() )

        copy_back_converged(centroids, opts, converged_d_centroids);
        gpuErrchk( cudaPeekAtLastError() )
        gpuErrchk( cudaDeviceSynchronize() )

        converged = true;
        for (int c = 0; c != opts->num_cluster; c++) {
            if (not centroids[c].converged) {
                converged = false;
                break;
            }
        }

        if (converged) break;
    }

    gettimeofday(&end, NULL);
    timing->total_iter_time = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * .001;
    timing->iter_to_converge = e;

    copy_back_data(points, centroids, opts, n_vals, coords_d_points, cluster_d_points, min_dist_d_points,
            coord_d_centroids, prev_coord_d_centroids, n_points_d_centroids, sums_d_centroids, converged_d_centroids);
    gpuErrchk( cudaPeekAtLastError() )
    gpuErrchk( cudaDeviceSynchronize() )

    free_device_mem(coords_d_points, cluster_d_points, min_dist_d_points,
            coord_d_centroids, prev_coord_d_centroids, n_points_d_centroids, sums_d_centroids, d_opts, d_n_vals);
    gpuErrchk( cudaPeekAtLastError() )
    gpuErrchk( cudaDeviceSynchronize() )
}
