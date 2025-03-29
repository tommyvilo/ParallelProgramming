#pragma once

#include <vector>
#include <random>
#include <iterator>
#include <algorithm>

template<typename T>
std::vector<T> random_square_matrix(int N, unsigned int seed); 

#ifdef PPROG_IMPLEMENTATION

template<>
std::vector<int> random_square_matrix<int>(int N, unsigned int seed) {

    std::minstd_rand engine(seed);
    std::uniform_int_distribution<int> distr(1, 100);

    std::vector<int> result;
    result.resize(N*N);
    std::generate(result.begin(), result.end(), [&]() {
        return distr(engine);
    });

    return result;
}

template<>
std::vector<float> random_square_matrix<float>(int N, unsigned int seed) {
    std::minstd_rand engine(seed);
    std::uniform_real_distribution<float> distr(0.0, 100.0);

    std::vector<float> result;
    result.resize(N*N);
    std::generate(result.begin(), result.end(), [&]() {
        return distr(engine);
    });

    return result;
}

#endif