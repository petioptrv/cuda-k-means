//
// Created by petioptrv on 2020-10-10.
//

#ifndef K_MEANS_HELPERS_H
#define K_MEANS_HELPERS_H

#include <math.h>
#include "centroid.h"
#include "point.h"
#include "argparse.h"

struct timing_t {
  int iter_to_converge;
  double total_iter_time;
};

void init_centroids(centroid_t **centroids, point_t *points, options_t *opts, int *n_vals);
void free_host_mem(options_t *opts, point_t *points, centroid_t *centroids);

void init_device_mem(point_t *points, centroid_t *centroids, options_t *opts, int *n_vals,
        double **coords_d_points, int **cluster_d_points, double **min_dist_d_points,
        double **coord_d_centroids, double **prev_coord_d_centroids,
        double **n_points_d_centroids, double **sums_d_centroids, bool **converged_d_centroids,
        options_t **d_opts, int **d_n_vals);
void copy_back_converged(centroid_t *centroids, options_t *opts, bool *converged_d_centroids);
void copy_back_data(point_t *points, centroid_t *centroids, options_t *opts, int *n_vals,
        double *coords_d_points, int *cluster_d_points, double *min_dist_d_points,
        double *coord_d_centroids, double *prev_coord_d_centroids,
        double *n_points_d_centroids, double *sums_d_centroids, bool *converged_d_centroids);
void free_device_mem(double *coords_d_points, int *cluster_d_points, double *min_dist_d_points,
        double *coord_d_centroids, double *prev_coord_d_centroids,
        double *n_points_d_centroids, double *sums_d_centroids,
        options_t *d_opts,  int *d_n_vals);

#endif //K_MEANS_HELPERS_H
