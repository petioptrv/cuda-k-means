//
// Created by petioptrv on 2020-10-05.
//

#ifndef K_MEANS_ARGPARSE_H
#define K_MEANS_ARGPARSE_H

#include <getopt.h>
#include <stdlib.h>
#include <iostream>

struct options_t {
  int num_cluster;
  int dims;
  char *inputfilename;
  int max_num_iter;
  double threshold;
  bool centroids;
  int seed;
  char algo;
  int thread_count;
};

void get_opts(int argc, char **argv, struct options_t *opts);

#endif //K_MEANS_ARGPARSE_H
