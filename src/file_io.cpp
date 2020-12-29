//
// Created by petioptrv on 2020-10-07.
//

#include "file_io.h"

void read_file(point_t **points, struct options_t* args, int *n_vals) {
    FILE *fp;
    int max_line_size = 1024;
    char line[max_line_size], *p, *e;
    double tmp_d;
    int base = 10;

    fp = fopen(args->inputfilename, "r");
    fgets(line, sizeof(line), fp);
    *n_vals = (int) strtol(line, &e, base);

    *points = (point_t*) malloc((*n_vals) * sizeof(point_t));

    for (int i = 0; i != *n_vals; i++) {
        auto *coord = (double*)malloc(args->dims * sizeof(double));

        fgets(line, sizeof(line), fp);
        p = line;
        (int)strtol(p, &e, base);  // consume first int

        for (int j = 0; j != args->dims; j++) {
            p = e;
            tmp_d = strtod(p, &e);
            coord[j] = tmp_d;
        }

        (*points)[i] = (point_t){coord, args->dims, -1, __DBL_MAX__};
    }
}
