#include "FuzzyKClusters.h"
#include <exception>
#include <iostream>
#include <cmath>


using namespace fkc;
using namespace Eigen;


// =============================================================================
// KMeans Algorithm
// =============================================================================

KMeans::KMeans()
: n_iters(0), variance(0.)
{
    // place holder
}

KMeans::~KMeans()
{
    // place holder
}

MatrixXd KMeans::Fit(const MatrixXd &data, int k, double q, double epsilon, int max_iters)
{
    auto res = Initialize(data, k, q);

    for (n_iters = 0; n_iters < max_iters; ++n_iters) {
        Distances(res, data);
        auto old_mems = mems;
        Memberships(q);
        FormClusters(res, data, q);

        if ((old_mems - mems).cwiseAbs().maxCoeff() < epsilon) {
            break;
        }
    }

    return res;
}

// initialize and guess the clusters
MatrixXd KMeans::Initialize(const MatrixXd &data, int k, double q)
{
    // resize matrices
    dists.resize(k, data.rows());

    // guess the cluster centers
    mems = MatrixXd::Random(k, data.rows());
    for (size_t j = 0; j < mems.cols(); ++j) {
        auto csum = mems.col(j).sum();
        for (size_t i = 0; i < mems.rows(); ++i) {
            mems(i, j) = mems(i, j)/csum;
        }
    }

    MatrixXd clusters(k, data.cols());
    FormClusters(clusters, data, q);
    return clusters;
}

// distance matrix (num_clusters, num_data)
void KMeans::Distances(const MatrixXd &centroids, const MatrixXd &data)
{
    for (size_t i = 0; i < centroids.rows(); ++i) {
        for (size_t j = 0; j < data.rows(); ++j) {
            dists(i, j) = std::sqrt((centroids.row(i) - data.row(j)).cwiseAbs2().sum());
        }
    }
}

// membership matrix (num_clusters, num_data)
void KMeans::Memberships(double q)
{
    // coeffcient-wise operation
    auto d = dists.array().pow(-2.0/(q - 1.0)).matrix();

    for (size_t j = 0; j < d.cols(); ++j) {
        auto dsum = d.col(j).sum();
        for (size_t i = 0; i < d.rows(); ++i) {
            mems(i, j) = d(i, j)/dsum;
        }
    }
}

// rebuild clusters
void KMeans::FormClusters(MatrixXd &clusters, const MatrixXd &data, double q)
{
    auto weights = mems.array().pow(q).matrix();
    for (size_t i = 0; i < clusters.rows(); ++i) {
        clusters.row(i) *= 0;
        for (size_t j = 0; j < data.rows(); ++j) {
            clusters.row(i) += data.row(j)*weights(i, j);
        }
        clusters.row(i) /= weights.row(i).sum();
    }
}


// =============================================================================
// KMeans Algorithm, extended for KRings
// =============================================================================
MatrixXd KRings::Fit(const MatrixXd &data, int k, double q, double epsilon, int max_iters)
{
    auto res = Initialize(data, k, q);

    for (n_iters = 0; n_iters < max_iters; ++n_iters) {
        Distances(res, data);
        auto old_mems = mems;
        Memberships(q);
        FormClusters(res, data, q);

        if ((old_mems - mems).cwiseAbs().maxCoeff() < epsilon) {
            break;
        }
    }

    return res;
}

// initialize and guess the clusters
MatrixXd KRings::Initialize(const MatrixXd &data, int k, double q)
{
    MatrixXd clusters(k, data.cols());
    FormClusters(clusters, data, q);
    return clusters;
}

// distance matrix (num_clusters, num_data)
void KRings::Distances(const MatrixXd &centroids, const MatrixXd &data)
{
}

// membership matrix (num_clusters, num_data)
void KRings::Memberships(double q)
{
}

// rebuild clusters
void KRings::FormClusters(MatrixXd &clusters, const MatrixXd &data, double q)
{
}

