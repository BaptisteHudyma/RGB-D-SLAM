#include "map_point.hpp"

#include "parameters.hpp"

namespace rgbd_slam {
    namespace utils {

        Point::Point (const vector3& coordinates, const cv::Mat& descriptor)
            : _coordinates(coordinates), _descriptor(descriptor)
        {}

        Map_Point::Map_Point(const vector3& coordinates, const cv::Mat& descriptor, const double observationTimeStamp) 
            : Point(coordinates, descriptor)
        {

            _age = 0;
            _lastUpdated =  observationTimeStamp;
            _counter = 0;
        }

        double Map_Point::get_liability() {
            double liability = _age / Parameters::get_point_age_liability();
            if (liability > 1.0)
                return 1.0;
            return liability;
        }

        /**
         * \brief True is this point is lost : should be removed from local map. Should be used only for map points
         */
        bool Map_Point::is_lost(const double currentTimeStamp) {
            return (_counter > Parameters::get_maximum_unmatched_before_removal()) ;//and ((currentTimeStamp - _lastUpdated) > MAX_LOST_TEMPORISATION);
        }

        /**
         * \brief Should add this staged point to the local map
         */
        bool Map_Point::should_add_to_map() {
            return (get_liability() > Parameters::get_minimum_liability_for_local_map());
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
        void Map_Point::update(const double observationTimeStamp, const vector3& newPointCoordinates, const cv::Mat& newDescriptor) 
        {
            _counter = 0;
            _age += 1;
            _coordinates = newPointCoordinates;
            _descriptor = newDescriptor;
            _lastUpdated = observationTimeStamp;
        }
    }
}
