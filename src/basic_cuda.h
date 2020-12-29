//
// Created by petioptrv on 2020-10-10.
//

#ifndef K_MEANS_BASIC_CUDA_H
#define K_MEANS_BASIC_CUDA_H

// wrapper CUDA function declared here

#include <cuda.h>
#include <sys/time.h>

#include "point.h"
#include "centroid.h"
#include "argparse.h"
#include "helpers.h"

void k_means_cuda_basic(point_t *points, centroid_t *centroids, options_t *opts, int *n_vals, timing_t *timing);
void k_means_cuda_smem(point_t *points, centroid_t *centroids, options_t *opts, int *n_vals, timing_t *timing);
void k_means_thrust(point_t *points, centroid_t *centroids, options_t *opts, int *n_vals, timing_t *timing);

#endif //K_MEANS_BASIC_CUDA_H
