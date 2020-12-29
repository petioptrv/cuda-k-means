//
// Created by petioptrv on 2020-10-07.
//

#include "point.h"

double distance(const double *coords1, const double *coords2, const int *dims) {
    double dist = 0;

    for (int i = 0; i != *dims; i++) {
        dist += (coords1[i] - coords2[i]) * (coords1[i] - coords2[i]);
    }

    return dist;
}
