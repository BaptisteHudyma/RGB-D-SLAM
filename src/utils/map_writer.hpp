#ifndef RGBDSLAM_UTILS_MAP_WRITER_HPP
#define RGBDSLAM_UTILS_MAP_WRITER_HPP

#include <fstream>
#include "types.hpp"

namespace rgbd_slam {
    namespace utils {

        class Map_Writer
        {
            public:
                Map_Writer(const std::string& filename);
                virtual ~Map_Writer();

                virtual void add_point(const vector3& pointCoordinates) = 0;

            protected:
                std::ofstream _file;

        };

        class XYZ_Map_Writer :
            public Map_Writer
        {
            public:
                XYZ_Map_Writer(const std::string& filename);

                void add_point(const vector3& pointCoordinates) override;
        };

        class PCD_Map_Writer:
            public Map_Writer
        {
            public:
                PCD_Map_Writer(const std::string& filename);

                void add_point(const vector3& pointCoordinates) override;
        };

    }   // utils
}   // rgbd_slam

#endif
