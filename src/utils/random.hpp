#ifndef RGBDSLAM_UTILS_RANDOM_HPP
#define RGBDSLAM_UTILS_RANDOM_HPP

#include <random>

namespace rgbd_slam { 
    namespace utils {


        class Random {
        public:
        /**
         * \brief Return a seeded random double between 0 and 1
         */
            static double get_random_double()
            {
                //static std::random_device randomDevice;
                static std::mt19937 randomEngine(0);
                static std::uniform_real_distribution<double> distribution(0.0, 1.0);
                
                return distribution(randomEngine);
            }

        };


    }
}



#endif