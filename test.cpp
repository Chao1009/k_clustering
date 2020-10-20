#include "FuzzyKClusters.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>


void block_assignment()
{
    Eigen::MatrixXd test(4, 3);
    test << 5, 5, 2,
            5, -5, 2,
            -5, -5, 2,
            -5, 5, 2;
    auto test2 = test.rightCols(1);
    test2 << 1, 1, 3, 3;
    std::cout << test << std::endl;
    std::cout << test2 << std::endl;
}

void test_fkm()
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

    std::cout << "FKM test: k input" << std::endl;
    std::cout << fkm.Fit(data, 4, 2.0, 1e-6) << std::endl;
    std::cout << fkm.NIters() << std::endl;
    std::cout << "FKM test: cluster center input" << std::endl;
    std::cout << fkm.Fit(data, centroids, 2.0, 1e-6) << std::endl;
    std::cout << fkm.NIters() << std::endl;
}

void test_fkr()
{
    std::default_random_engine rng;
    std::uniform_real_distribution<double> uni(0., 2.*M_PI);
    auto fkr = fkc::KRings();

    Eigen::MatrixXd centroids(4, 3);
    centroids << 5, 5, 3,
                 5, -5, 1,
                -5, -5, 2,
                -5, 5, 2;

    Eigen::MatrixXd data(80, 2);
    for (size_t i = 0; i < 4; ++i) {
        auto cx = centroids(i, 0);
        auto cy = centroids(i, 1);
        auto cr = centroids(i, 2);

        for (size_t j = 0; j < 20; ++j) {
            auto phi = uni(rng);
            data.row(i*20 + j) << cx + std::cos(phi)*cr, cy + std::sin(phi)*cr;
        }
    }

    std::cout << "FKR test: k input" << std::endl;
    auto fit = fkr.Fit(data, 4, 2.0, 1e-6);
    std::cout << fit << std::endl;
    std::cout << fkr.NIters() << ": " << fkr.Variance() << std::endl;
    std::cout << fkr.Quality(data, fit, fkr.GetMemberships()) << std::endl;

    std::cout << "FKR test: cluster center input" << std::endl;
    fit = fkr.Fit(data, centroids.leftCols(2), 2.0, 1e-6);
    std::cout << fit << std::endl;
    std::cout << fkr.NIters() << ": " << fkr.Variance() << std::endl;
    std::cout << fkr.Quality(data, fit, fkr.GetMemberships()) << std::endl;
}

int main(int argc, char *argv[])
{
    // block_assignment();
    // test_fkm();
    test_fkr();
    return 0;
}

