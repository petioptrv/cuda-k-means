//
// Created by petioptrv on 2020-10-10.
//

#ifndef K_MEANS_CENTROID_H
#define K_MEANS_CENTROID_H

struct centroid_t {
  double *coord;
  double *prev_coord;
  int n_points;
  double *sums;
  bool converged;
};

#endif //K_MEANS_CENTROID_H
