#ifndef RGBDSLAM_UTILS_RANDOM_HPP
#define RGBDSLAM_UTILS_RANDOM_HPP

#include <ctime>
#include <random>

namespace rgbd_slam::utils {

class Random
{
  public:
    /**
     * \brief Return a seeded random double between 0 and 1, around a uniform distribution
     */
    static double get_random_double()
    {
        thread_local std::mt19937 randomEngine(_seed);
        thread_local std::uniform_real_distribution distribution(0.0, 1.0);

        return distribution(randomEngine);
    }

    /**
     * \brief Return a seeded random double between -1 and 1, around a normal distribution
     */
    static double get_normal_double()
    {
        thread_local std::mt19937 randomEngine(_seed);
        thread_local std::normal_distribution distribution(0.0, 1.0);

        return distribution(randomEngine);
    }

    static uint get_random_uint(const uint maxValue) { return static_cast<uint>(maxValue * get_random_double()); }

#ifndef MAKE_DETERMINISTIC
    inline static const uint _seed = std::time(0);
#else
    // whatever seed
    inline static const uint _seed = 0;
#endif
};

} // namespace rgbd_slam::utils

#endif