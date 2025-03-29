#pragma once

#include <limits>

bool float_approx_equal(float a, float b, float epsilon = std::numeric_limits<float>::epsilon());

#ifdef PPROG_IMPLEMENTATION

bool float_approx_equal(float a, float b, float epsilon) {
    return fabs(a - b) <= ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

#endif