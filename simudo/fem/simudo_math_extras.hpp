// Created by Claude Sonnet 4.6 (Anthropic), 2026-05-14
#pragma once
#include <boost/math/special_functions/lambert_w.hpp>
#include <cmath>

// Safe wrapper: boost::math::lambert_w0 throws std::overflow_error for
// non-finite inputs. We return x directly so that inf -> inf, nan -> nan.
inline double simudo_safe_lambert_w0(double x) {
    if (!std::isfinite(x)) return x;
    return boost::math::lambert_w0(x);
}
