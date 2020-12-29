#include <iostream>

#include "file_io.h"
#include "sequential.h"
#include "helpers.h"
#include "basic_cuda.h"

void printouts(timing_t *timing, options_t *opts, point_t *points, int *n_vals, centroid_t *centroids) {
    printf("%d, %lf\n", timing->iter_to_converge, timing->total_iter_time / timing->iter_to_converge);

    if (not opts->centroids) {
        printf("clusters:");
        for (int p = 0; p != *n_vals; ++p) {
            printf(" %d", points[p].cluster);
        }
    } else {
        for (int c = 0; c != opts->num_cluster; ++c) {
            printf("%d ", c);
            for (int d = 0; d != opts->dims; ++d) {
                printf("%lf ", centroids[c].coord[d]);
            }
            printf("\n");
        }
    }
}

int main(int argc, char **argv) {
    auto* opts = (options_t *) malloc(sizeof(options_t));
    int n_vals;
    point_t *points;
    centroid_t *centroids;
    timing_t timing = {0, 0};

    get_opts(argc, argv, opts);
    read_file(&points, opts, &n_vals);
    init_centroids(&centroids, points, opts, &n_vals);

    switch (opts->algo) {
    case 's':
        k_means_seq(points, centroids, opts, &n_vals, &timing);
        break;
    case 'b':
        k_means_cuda_basic(points, centroids, opts, &n_vals, &timing);
        break;
    case 't':
        fprintf(stderr, "Thrust not implemented.");
        exit(ENOTSUP);
    case 'm':
        k_means_cuda_smem(points, centroids, opts, &n_vals, &timing);
        break;
    default:
        fprintf(stderr, "Valid algorithm values are s, t, b, m.");
        exit(ENOTSUP);
    }

    printouts(&timing, opts, points, &n_vals, centroids);

    free_host_mem(opts, points, centroids);
}
