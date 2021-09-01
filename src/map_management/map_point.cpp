#include "map_point.hpp"

//Observe a point for N frames to gain max liability
#define MAP_POINT_AGE_LIABILITY 5
// Maximum time before we consider this point too old
#define MAX_LOST_TEMPORISATION 1
// Minimum point liability for the local map
#define MINIMUM_LIABILITY_FOR_LOCAL_MAP 0.7
// Max unmatched points to consider this map point as lost
#define MAX_UNMTACHED_FOR_TRACKING 5

namespace rgbd_slam {
    namespace utils {

        Point::Point (const vector3& coordinates, const cv::Mat& descriptor)
            : _coordinates(coordinates), _descriptor(descriptor)
        {}

        Map_Point::Map_Point(const vector3& coordinates, const cv::Mat& descriptor, double observationTimeStamp) 
            : Point(coordinates, descriptor)
        {

            _age = 0;
            _lastUpdated =  observationTimeStamp;
            _counter = 0;
        }

        double Map_Point::get_liability() {
            double liability = _age /  MAP_POINT_AGE_LIABILITY;
            if (liability > 1.0)
                return 1.0;
            return liability;
        }

        /**
         * \brief True is this point is lost : should be removed from local map. Should be used only for map points
         */
        bool Map_Point::is_lost(double currentTimeStamp) {
            return (_counter > MAX_UNMTACHED_FOR_TRACKING) ;//and ((currentTimeStamp - _lastUpdated) > MAX_LOST_TEMPORISATION);
        }

        /**
         * \brief Should add this staged point to the local map
         */
        bool Map_Point::should_add_to_map() {
            return (get_liability() > MINIMUM_LIABILITY_FOR_LOCAL_MAP);
        }

        /**
         * \brief Update this point without it being detected/matched
         */
        void Map_Point::update_unmacthed()
        {
            // decrease successful matches
            if (_age > 0)
                _age -= 1;
            _counter += 1;
        }

        /**
         * \brief Update this map point with the given informations: it is matched with another point
         */
        void Map_Point::update(double observationTimeStamp, const vector3& newPointCoordinates, const cv::Mat& newDescriptor) 
        {
            _counter = 0;
            _age += 1;
            _coordinates = newPointCoordinates;
            _descriptor = newDescriptor;
            _lastUpdated = observationTimeStamp;
        }
    }
}
