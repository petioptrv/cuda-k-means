cmake_minimum_required(VERSION 3.16)
project(k_means LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

add_executable(k_means
        src/argparse.cpp src/argparse.h
        src/basic_cuda.cu src/basic_cuda.h
        src/centroid.h
        src/file_io.cpp src/file_io.h
        src/helpers.cpp src/helpers.h
        src/main.cpp
        src/point.cpp src/point.h
        src/sequential.cpp src/sequential.h
        src/smem_cuda.cu
        src/thrust.cu)

set_target_properties(
        k_means
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)