//
// Created by petioptrv on 2020-10-07.
//

#ifndef K_MEANS_FILE_IO_H
#define K_MEANS_FILE_IO_H

#include <stdlib.h>
#include <fstream>
#include <iostream>

#include "point.h"
#include "argparse.h"

void read_file(point_t **points, struct options_t* args, int *n_vals);

#endif //K_MEANS_FILE_IO_H
