#ifndef RGBDSLAM_UTILS_RANDOM_HPP
#define RGBDSLAM_UTILS_RANDOM_HPP

#include <Eigen/src/Core/Matrix.h>
#include <cassert>
#include <ctime>
#include <random>

namespace rgbd_slam::utils {

/**
 * \brief static class, to handle the random generation
 */
class Random
{
  public:
    [[nodiscard]] static std::mt19937& get_random_engine()
    {
        static thread_local std::mt19937 randomEngine(_seed);
        return randomEngine;
    }

    /**
     * \brief Return a seeded random double between 0 and 1, around a uniform distribution
     */
    [[nodiscard]] static double get_random_double()
    {
        thread_local std::uniform_real_distribution distribution(0.0, 1.0);
        return distribution(get_random_engine());
    }

    /**
     * \brief Return a seeded random double between -1 and 1, around a normal distribution
     */
    [[nodiscard]] static double get_normal_double()
    {
        thread_local std::normal_distribution distribution(0.0, 1.0);
        return distribution(get_random_engine());
    }

    template<int Size> [[nodiscard]] static Eigen::Vector<double, Size> get_normal_doubles()
    {
        Eigen::Vector<double, Size> vec;
        for (uint i = 0; i < Size; ++i)
            vec(i) = get_normal_double();
        return vec;
    }

    /**
     * \brief Return a random int in the [minValue, maxValue[ interval
     */
    [[nodiscard]] static uint get_random_uint(const uint minValue, const uint maxValue)
    {
        assert(minValue < maxValue);
        return minValue + static_cast<uint>(std::floor(get_random_double() * (maxValue - minValue)));
    }

    [[nodiscard]] static uint get_random_uint(const uint maxValue) { return get_random_uint(0, maxValue); }

#ifndef MAKE_DETERMINISTIC
    inline static const uint _seed = std::time(0);
#else
    // whatever seed
    inline static const uint _seed = 0;
#endif
};

} // namespace rgbd_slam::utils

#endif