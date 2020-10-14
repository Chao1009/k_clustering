#include "FuzzyKClusters.h"
#include <iostream>
#include <random>


int main(int argc, char *argv[])
{
    std::default_random_engine rng;
    std::uniform_real_distribution<double> uni(-2.0, 2.0);
    auto fkm = fkc::KMeans();

    Eigen::MatrixXd centroids(4, 2);
    centroids << 5, 5,
                 5, -5,
                -5, -5,
                -5, 5;

    Eigen::MatrixXd data(200, 2);
    for (size_t i = 0; i < 4; ++i) {
        auto cx = centroids(i, 0);
        auto cy = centroids(i, 1);

        for (size_t j = 0; j < 50; ++j) {
            data.row(i*50 + j) << cx + uni(rng), cy + uni(rng);
        }
    }

    std::cout << fkm.Fit(data, 4, 2.0, 1e-6) << std::endl;
    std::cout << fkm.NIters() << std::endl;
    return 0;
}
