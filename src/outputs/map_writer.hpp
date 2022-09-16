#ifndef RGBDSLAM_OUTPUTS_MAP_WRITER_HPP
#define RGBDSLAM_OUTPUTS_MAP_WRITER_HPP

#include <fstream>
#include "../types.hpp"

namespace rgbd_slam {
    namespace outputs {

        class IMap_Writer
        {
            public:
                IMap_Writer(const std::string& filename);
                virtual ~IMap_Writer();

                virtual void add_point(const vector3& pointCoordinates) = 0;

            protected:
                std::ofstream _file;

        };

        class XYZ_Map_Writer :
            public IMap_Writer
        {
            public:
                explicit XYZ_Map_Writer(const std::string& filename);

                void add_point(const vector3& pointCoordinates) override;
        };

        class PCD_Map_Writer:
            public IMap_Writer
        {
            public:
                explicit PCD_Map_Writer(const std::string& filename);

                void add_point(const vector3& pointCoordinates) override;
        };

    }   // outputs
}   // rgbd_slam

#endif
