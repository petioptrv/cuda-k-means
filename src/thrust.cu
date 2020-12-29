
#include "basic_cuda.h"

#include <thrust/device_vector.h>

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/iterator_adaptor.h>

// derive repeat_iterator from iterator_adaptor
template<typename Iterator>
class fancy_repeat_iterator
        : public thrust::iterator_adaptor<
                fancy_repeat_iterator<Iterator>, // the first template parameter is the name of the iterator we're creating
                Iterator                   // the second template parameter is the name of the iterator we're adapting
                // we can use the default for the additional template parameters
        >
{
public:
    // shorthand for the name of the iterator_adaptor we're deriving from
    typedef thrust::iterator_adaptor<fancy_repeat_iterator<Iterator>, Iterator> super_t;
    __host__ __device__
    fancy_repeat_iterator(const Iterator &x, int n, int stride) : super_t(x), begin(x), n(n), stride(stride) {}
    // befriend thrust::iterator_core_access to allow it access to the private interface below
    friend class thrust::iterator_core_access;
private:
    // repeat each sequence of elements of the adapted range n times
    unsigned int n;
    // sequence length
    unsigned int stride;
    // used to keep track of where we began
    const Iterator begin;
    // it is private because only thrust::iterator_core_access needs access to it
    __host__ __device__
    typename super_t::reference dereference() const
    {
        unsigned int base = (int)*this->base();
        return *(begin + (this->base() - begin) / (n * stride) * stride + (base % stride));
    }
};

// derive repeat_iterator from iterator_adaptor
template<typename Iterator>
class loop_back_iterator
        : public thrust::iterator_adaptor<
                loop_back_iterator<Iterator>, // the first template parameter is the name of the iterator we're creating
                Iterator                   // the second template parameter is the name of the iterator we're adapting
                // we can use the default for the additional template parameters
        >
{
public:
    // shorthand for the name of the iterator_adaptor we're deriving from
    typedef thrust::iterator_adaptor<loop_back_iterator<Iterator>, Iterator> super_t;
    __host__ __device__
    loop_back_iterator(const Iterator &x, int n) : super_t(x), begin(x), n(n) {}
    // befriend thrust::iterator_core_access to allow it access to the private interface below
    friend class thrust::iterator_core_access;
private:
    // repeat each element of the adapted range n times
    unsigned int n;
    // used to keep track of where we began
    const Iterator begin;
    // it is private because only thrust::iterator_core_access needs access to it
    __host__ __device__
    typename super_t::reference dereference() const
    {
        printf("\nlb: %d", *(begin + (this->base() - begin) % n));
        return *(begin + (this->base() - begin) % n);
    }
};


struct L2 {
  L2() {}
  __host__ __device__
  double operator() (const double &x, const double &y) {
      double val = (x - y) * (x - y);
      printf("\n%f", val);
      return (x - y) * (x - y);
  }
};

// derive repeat_iterator from iterator_adaptor
template<typename Iterator>
class repeat_iterator
        : public thrust::iterator_adaptor<
                repeat_iterator<Iterator>, // the first template parameter is the name of the iterator we're creating
                Iterator                   // the second template parameter is the name of the iterator we're adapting
                // we can use the default for the additional template parameters
        >
{
public:
    // shorthand for the name of the iterator_adaptor we're deriving from
    typedef thrust::iterator_adaptor<
            repeat_iterator<Iterator>,
            Iterator
    > super_t;
    __host__ __device__
    repeat_iterator(const Iterator &x, int n) : super_t(x), begin(x), n(n) {}
    // befriend thrust::iterator_core_access to allow it access to the private interface below
    friend class thrust::iterator_core_access;
private:
    // repeat each element of the adapted range n times
    unsigned int n;
    // used to keep track of where we began
    const Iterator begin;
    // it is private because only thrust::iterator_core_access needs access to it
    __host__ __device__
    typename super_t::reference dereference() const
    {
        return *(begin + (this->base() - begin) / n);
    }
};

void k_means_thrust(point_t *points, centroid_t *centroids, options_t *opts, int *n_vals, timing_t *timing) {
//    thrust::counting_iterator<int> count0(0);
//    loop_back_iterator<thrust::counting_iterator<int>> test(count0, 6);
//
//    for (int i = 0; i != 20; i++) {
//        std::cout << test[i] << std::endl;
//    }
//
//    return;


    int n_points = *n_vals;
    int n_clusters = opts->num_cluster;
    int n_dims = opts->dims;
    int t_data_pts = n_points * n_clusters * n_dims;

    // INITIALIZATION
    thrust::device_vector<double> coords_d_points(n_points * n_dims);
    thrust::device_vector<int> cluster_d_points(n_points);
    thrust::device_vector<double> min_dist_d_points(n_points);

    thrust::device_vector<double> coord_d_centroids(n_clusters * n_dims);
    thrust::device_vector<double> prev_coord_d_centroids(n_clusters * n_dims);
    thrust::device_vector<double> n_points_d_centroids(n_clusters);
    thrust::device_vector<double> sums_d_centroids(n_clusters * n_dims);

    for (int p = 0; p != n_points; p++) {
        thrust::copy_n(points[p].coord, n_dims, coords_d_points.begin() + p * n_dims);
    }
    for (int c = 0; c != n_clusters; c++) {
        thrust::copy_n(centroids[c].coord, n_dims, coord_d_centroids.begin() + c * n_dims);
    }

    thrust::device_vector<double> temp(t_data_pts);
    thrust::device_vector<int> temp_idx(t_data_pts);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    bool converged = false;
    int e;
    for (e = 0; e != opts->max_num_iter or converged; e++) {

        thrust::fill(n_points_d_centroids.begin(), n_points_d_centroids.end(), (double)0);
        thrust::fill(sums_d_centroids.begin(), sums_d_centroids.end(), (double)0);
        thrust::fill(min_dist_d_points.begin(), min_dist_d_points.end(), __DBL_MAX__);

        // ITERATORS
        thrust::counting_iterator<long> count1(0);
        loop_back_iterator<thrust::counting_iterator<long>> cluster_coord_iter(count1, n_clusters * n_dims);
        thrust::counting_iterator<int> count2(0);
        fancy_repeat_iterator<thrust::counting_iterator<int>> points_coord_iter(count2, n_clusters, n_dims);  // this works
        thrust::counting_iterator<int> count3(0);
        repeat_iterator<thrust::counting_iterator<int>> dist_sum_iterator(count3, n_dims);

        auto mapping_cluster_coords = thrust::make_permutation_iterator(
                coord_d_centroids.begin(), cluster_coord_iter.base()
        );
        auto mapping_points_coords = thrust::make_permutation_iterator(
                coords_d_points.begin(), points_coord_iter.base()
        );

        for (int i = n_dims * 0; i != n_dims * 0 + n_dims; i++ ) {
            std::cout << mapping_cluster_coords[i] << std::endl;
        }

//        thrust::transform(mapping_points_coords.base(), mapping_cluster_coords.base() + n_dims,
//                mapping_points_coords.base(), temp.begin(), L2());

        // sum per point coords
        thrust::reduce_by_key(
                dist_sum_iterator.base(),
                dist_sum_iterator.base() + t_data_pts,
                temp.begin(),
                temp_idx.begin(),
                temp.begin()
                );
        // pick min per cluster count

    }

    gettimeofday(&end, NULL);
    timing->total_iter_time = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * .001;
    timing->iter_to_converge = e;

    const int N = 7;
    int A[N] = {1, 3, 3, 3, 1, 1, 3}; // input keys
    int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
    int C[N];                         // output keys
    int D[N];                         // output values
    thrust::pair<int*,int*> new_end;
    new_end = thrust::reduce_by_key(A, A + N, B, A, B);

    for (int i = 0; i != N; i++) {
        printf("%d ", A[i]);
    }
    printf("\n");
    for (int i = 0; i != N; i++) {
        printf("%d ", B[i]);
    }
    printf("\n");

    return;
}
