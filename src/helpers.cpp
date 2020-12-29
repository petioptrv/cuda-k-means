//
// Created by petioptrv on 2020-10-10.
//

#include "basic_cuda.h"
#include <cuda_runtime.h>
#include "helpers.h"

void init_centroids(centroid_t **centroids, point_t *points, options_t *opts, int *n_vals) {
    *centroids = (centroid_t*) malloc(opts->num_cluster * sizeof(centroid_t));

    // init
    for (int i = 0; i != opts->num_cluster; ++i) {
        (*centroids)[i] = {
                (double*)malloc(opts->dims * sizeof(double)),
                (double*)malloc(opts->dims * sizeof(double)),
                0,
                (double*)malloc(opts->dims * sizeof(double)),
                false
        };
    }

    // randomize
    static unsigned long int next = opts->seed;
    static unsigned long kmeans_rmax = 32767;
    for (int c = 0; c != opts->num_cluster; ++c) {
        next = next * 1103515245 + 12345;
        int rand = (unsigned int)(next/65536) % (kmeans_rmax + 1);
        int index = rand % *n_vals;
        for (int d = 0; d != opts->dims; ++d) {
            (*centroids)[c].coord[d] = points[index].coord[d];
        }
    }
}

void free_host_mem(options_t *opts, point_t *points, centroid_t *centroids) {
    free(opts);
    free(points);
    free(centroids);
}

void init_device_mem(point_t *points, centroid_t *centroids, options_t *opts, int *n_vals,
        double **coords_d_points, int **cluster_d_points, double **min_dist_d_points,
        double **coord_d_centroids, double **prev_coord_d_centroids,
        double **n_points_d_centroids, double **sums_d_centroids, bool **converged_d_centroids,
        options_t **d_opts, int **d_n_vals) {

    long coord_size = opts->dims * sizeof(double);
    long all_coord_points_size = *n_vals * coord_size;
    int all_cluster_size = *n_vals * sizeof(int);
    long all_min_dist_size = *n_vals * sizeof(double);
    long all_coord_clusterss_size = opts->num_cluster * coord_size;
    int all_n_points_size = opts->num_cluster * sizeof(double);
    int all_bool_size = opts->num_cluster * sizeof(bool);
    int opts_size = sizeof(options_t);
    int n_vals_size = sizeof(int);

    cudaMalloc((void **)coords_d_points, all_coord_points_size);
    cudaMalloc((void **)cluster_d_points, all_cluster_size);
    cudaMalloc((void **)min_dist_d_points, all_min_dist_size);
    cudaMalloc((void **)coord_d_centroids, all_coord_clusterss_size);
    cudaMalloc((void **)prev_coord_d_centroids, all_coord_clusterss_size);
    cudaMalloc((void **)n_points_d_centroids, all_n_points_size);
    cudaMalloc((void **)sums_d_centroids, all_coord_clusterss_size);
    cudaMalloc((void **)converged_d_centroids, all_bool_size);
    cudaMalloc((void **)d_opts, opts_size);
    cudaMalloc((void **)d_n_vals, n_vals_size);

    for (int i = 0; i != *n_vals; i++) {
        cudaMemcpy((*coords_d_points) + i * opts->dims, points[i].coord, coord_size, cudaMemcpyHostToDevice);
        cudaMemcpy((*cluster_d_points) + i, &points[i].cluster, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy((*min_dist_d_points) + i, &points[i].min_dist, sizeof(double), cudaMemcpyHostToDevice);
    }

    double n_points_centroid;
    for (int d = 0; d != opts->num_cluster; d++) {
        cudaMemcpy((*coord_d_centroids) + d * opts->dims, centroids[d].coord, coord_size, cudaMemcpyHostToDevice);
        cudaMemcpy((*prev_coord_d_centroids) + d * opts->dims, centroids[d].prev_coord, coord_size, cudaMemcpyHostToDevice);
        n_points_centroid = centroids[d].n_points;
        cudaMemcpy((*n_points_d_centroids) + d, &n_points_centroid, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy((*sums_d_centroids) + d * opts->dims, centroids[d].sums, coord_size, cudaMemcpyHostToDevice);
        cudaMemcpy((*converged_d_centroids) + d, &centroids[d].converged, sizeof(bool), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(*d_opts, opts, opts_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_n_vals, n_vals, n_vals_size, cudaMemcpyHostToDevice);
}

void copy_back_converged(centroid_t *centroids, options_t *opts, bool *converged_d_centroids) {
    for (int d = 0; d != opts->num_cluster; d++) {
        cudaMemcpy(&centroids[d].converged, converged_d_centroids + d, sizeof(bool), cudaMemcpyDeviceToHost);
    }
}

void copy_back_data(point_t *points, centroid_t *centroids, options_t *opts, int *n_vals,
        double *coords_d_points, int *cluster_d_points, double *min_dist_d_points,
        double *coord_d_centroids, double *prev_coord_d_centroids,
        double *n_points_d_centroids, double *sums_d_centroids, bool *converged_d_centroids) {
    long coord_size = opts->dims * sizeof(double);

    for (int i = 0; i != *n_vals; i++) {
        cudaMemcpy(points[i].coord, coords_d_points + i * opts->dims, coord_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(&points[i].cluster, cluster_d_points + i, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&points[i].min_dist, min_dist_d_points + i, sizeof(double), cudaMemcpyDeviceToHost);
    }

    auto *n_points_centroid = (double *)malloc(opts->num_cluster * sizeof(double));
    cudaMemcpy(n_points_centroid, n_points_d_centroids, opts->num_cluster * sizeof(double), cudaMemcpyDeviceToHost);
    for (int d = 0; d != opts->num_cluster; d++) {
        cudaMemcpy(centroids[d].coord, coord_d_centroids + d * opts->dims, coord_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(centroids[d].prev_coord, prev_coord_d_centroids + d * opts->dims, coord_size, cudaMemcpyDeviceToHost);
        centroids[d].n_points = (int)n_points_centroid[d];
        cudaMemcpy(centroids[d].sums, sums_d_centroids + d * opts->dims, coord_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(&centroids[d].converged, converged_d_centroids + d, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    free(n_points_centroid);
}

void free_device_mem(double *coords_d_points, int *cluster_d_points, double *min_dist_d_points,
        double *coord_d_centroids, double *prev_coord_d_centroids,
        double *n_points_d_centroids, double *sums_d_centroids,
        options_t *d_opts,  int *d_n_vals) {
    cudaFree(coords_d_points);
    cudaFree(cluster_d_points);
    cudaFree(min_dist_d_points);
    cudaFree(coord_d_centroids);
    cudaFree(prev_coord_d_centroids);
    cudaFree(n_points_d_centroids);
    cudaFree(sums_d_centroids);
    cudaFree(d_opts);
    cudaFree(d_n_vals);
}