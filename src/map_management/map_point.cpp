#include "map_point.hpp"

#include "parameters.hpp"

namespace rgbd_slam {
    namespace map_management {

        Point::Point (const vector3& coordinates, const cv::Mat& descriptor)
            : _coordinates(coordinates), _descriptor(descriptor)
        {}


        /**
         *      Staged_Point
         */

        Staged_Point::Staged_Point(const vector3& coordinates, const cv::Mat& descriptor)
            : Point(coordinates, descriptor)
        {
            _matchesCount = 0; 
        }

        double Staged_Point::get_confidence() const 
        {
            const double confidence = static_cast<double>(_matchesCount) / static_cast<double>(Parameters::get_point_staged_age_confidence());
            return std::max(std::min(confidence, 1.0), -1.0);
        }

        bool Staged_Point::should_add_to_local_map() const
        {
            return (get_confidence() > Parameters::get_minimum_confidence_for_local_map());
        }


        void Staged_Point::update_unmatched(int removeNMatches)
        {
            _matchesCount -= removeNMatches;
        }

        void Staged_Point::update_matched(const vector3& newPointCoordinates, const cv::Mat& newDescriptor)
        {
            _matchesCount += 1;
            // TODO: update coordinates with error
            _descriptor = newDescriptor;
        }

        bool Staged_Point::should_remove_from_staged() const
        {
            return _matchesCount < 0;
        }

        /**
         *     Map_Point 
         */


        Map_Point::Map_Point(const vector3& coordinates, const cv::Mat& descriptor) 
            : Point(coordinates, descriptor)
        {

            _age = 0;
            _failTrackingCount = 0;
        }

        double Map_Point::get_confidence() const
        {
            double confidence = static_cast<double>(_age) / static_cast<double>(Parameters::get_point_age_confidence());
            return std::max(std::min(confidence, 1.0), -1.0);
        }

        /**
         * \brief True is this point is lost : should be removed from local map. Should be used only for map points
         */
        bool Map_Point::is_lost() const {
            return (_failTrackingCount > Parameters::get_maximum_unmatched_before_removal());
        }

        /**
         * \brief Update this point without it being detected/matched
         */
        void Map_Point::update_unmatched()
        {
            // decrease successful matches
            _age -= 1;
            _failTrackingCount += 1;
        }

        /**
         * \brief Update this map point with the given informations: it is matched with another point
         */
        double Map_Point::update_matched(const vector3& newPointCoordinates, const cv::Mat& newDescriptor) 
        {
            _failTrackingCount = 0;
            _age += 1;
            // TODO: update coordinates with error
            //_coordinates = newPointCoordinates;
            _descriptor = newDescriptor;

            return    
                abs(newPointCoordinates.x() - _coordinates.x()) + 
                abs(newPointCoordinates.y() - _coordinates.y()) +
                abs(newPointCoordinates.z() - _coordinates.z());
        }
    }   /* map_management */
} /* rgbd_slam */
