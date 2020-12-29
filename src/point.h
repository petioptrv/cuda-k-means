//
// Created by petioptrv on 2020-10-07.
//

#ifndef K_MEANS_POINT_H
#define K_MEANS_POINT_H

struct point_t {
  double *coord;
  int dims;
  int cluster;
  double min_dist;
};

double distance(const double *coords1, const double *coords2, const int *dims);

#endif //K_MEANS_POINT_H
