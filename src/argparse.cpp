//
// Created by petioptrv on 2020-10-05.
//

#include "argparse.h"

void get_opts(int argc, char **argv, struct options_t *opts) {
    if (argc == 1) {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t-k <num_cluster>" << std::endl;
        std::cout << "\t-d <dims>" << std::endl;
        std::cout << "\t-i <inputfilename>" << std::endl;
        std::cout << "\t-m <max_num_iter>" << std::endl;
        std::cout << "\t-t <threshold>" << std::endl;
        std::cout << "\t-s <seed>" << std::endl;
        std::cout << "\t-a <algo> one of s <sequential>, b <basic CUDA>, t <Thrust>, m <CUDA Shmem>" << std::endl;
        std::cout << "\t[Optional] -c output centroids of all clusters" << std::endl;
        exit(0);
    }

    opts->centroids = false;

    struct option l_opts[] = {
            {"num_cluster", required_argument, NULL, 'k'},
            {"dims", required_argument, NULL, 'd'},
            {"inputfilename", required_argument, NULL, 'i'},
            {"max_num_iter", required_argument, NULL, 'm'},
            {"threshold", required_argument, NULL, 't'},
            {"seed", required_argument, NULL, 's'},
            {"algo", required_argument, NULL, 'a'},
            {"centroids", no_argument, NULL, 'c'},
            {"thread_count", required_argument, NULL, 'p'},
    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "k:d:i:m:t:cs:a:p:", l_opts, &ind)) != -1) {
        switch (c) {
        case 0:
            break;
        case 'k':
            opts->num_cluster = atoi((char *)optarg);
            break;
        case 'd':
            opts->dims = atoi((char *)optarg);
            break;
        case 'i':
            opts->inputfilename = (char *)optarg;
            break;
        case 'm':
            opts->max_num_iter = atoi((char *)optarg);
            break;
        case 't':
            opts->threshold = atof((char *)optarg);
            break;
        case 'c':
            opts->centroids = true;
            break;
        case 's':
            opts->seed = atof((char *)optarg);
            break;
        case 'a':
            opts->algo = (char)*optarg;
            break;
        case 'p':
            opts->thread_count = atoi((char *)optarg);
            break;
        case ':':
            fprintf(stderr, "%s: option -%c requires an argument.\n", argv[0], (char)optopt);
            exit(EXIT_FAILURE);
        }
    }
}
