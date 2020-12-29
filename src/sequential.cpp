//
// Created by petioptrv on 2020-10-05.
//

#include "sequential.h"

void k_means_seq(point_t *points, centroid_t *centroids, options_t *opts, int *n_vals, timing_t *timing) {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    int e;
    for (e = 0; e != opts->max_num_iter; ++e) {
        // RESET VALUES
        for (int c = 0; c != opts->num_cluster; ++c) {
            centroids[c].n_points = 0;
            for (int d = 0; d != opts->dims; ++d) {
                centroids[c].sums[d] = 0;
                centroids[c].prev_coord[d] = centroids[c].coord[d];
            }
        }

        // ASSIGN POINTS TO CLUSTERS
        for (int p = 0; p != *n_vals; ++p) {
            point_t *point = &(points[p]);
            centroid_t *centroid;

            points[p].min_dist = __DBL_MAX__;

            for (int c = 0; c != opts->num_cluster; ++c) {
                centroid = &(centroids[c]);
                double dist = distance(point->coord, centroid->coord, &(opts->dims));
                if (dist < point->min_dist) {
                    point->min_dist = dist;
                    point->cluster = c;
                }
            }

            centroid = &(centroids[point->cluster]);
            centroid->n_points++;
            for (int d = 0; d != opts->dims; ++d) {
                centroid->sums[d] += point->coord[d];
            }
        }

        // COMPUTING NEW CENTROIDS
        bool converged = true;

        for (int c = 0; c != opts->num_cluster; ++c) {
            centroid_t *centroid = &(centroids[c]);
            for (int d = 0; d != opts->dims; ++d) {
                centroid->coord[d] = centroid->sums[d] / centroid->n_points;
            }

            if (converged) {
                double delta = sqrt(distance(centroid->prev_coord, centroid->coord, &(opts->dims)));
                if (delta > opts->threshold) {
                    converged = false;
                }
            }
        }

        // MAYBE STOP
        if (converged) {
            break;
        }
    }

    gettimeofday(&end, NULL);
    timing->total_iter_time = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * .001;
    timing->iter_to_converge = e;
}
