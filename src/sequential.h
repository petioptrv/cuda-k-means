//
// Created by petioptrv on 2020-10-05.
//

#ifndef K_MEANS_SEQUENTIAL_H
#define K_MEANS_SEQUENTIAL_H

#include <math.h>
#include <sys/time.h>

#include "point.h"
#include "argparse.h"
#include "centroid.h"
#include "helpers.h"

void k_means_seq(point_t *points, centroid_t *centroids, options_t *opts, int *n_vals, timing_t *timing);

#endif //K_MEANS_SEQUENTIAL_H
