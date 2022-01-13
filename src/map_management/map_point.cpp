#include "map_point.hpp"
#include "parameters.hpp"

#include <Eigen/LU>

namespace rgbd_slam {
    namespace map_management {

        const double useOfNewPosition = 0.1;    //[0 ; 1] 0 will ignore the new value, 1 will replace the current value with the new one

        Point::Point (const vector3& coordinates, const cv::Mat& descriptor) :
            _coordinates(coordinates), 
            _descriptor(descriptor),
            _id(Point::_currentPointId)
        {
            Point::_currentPointId += 1;
            _screenCoordinates = cv::Point2f(-1, -1);
        }

        Point::Point (const vector3& coordinates, const cv::Mat& descriptor, const size_t id) :
            _coordinates(coordinates), 
            _descriptor(descriptor),
            _id(id)
        {
            _screenCoordinates = cv::Point2f(-1, -1);
        }

        /**
         *     Tracked point
         */

        IMap_Point_With_Tracking::IMap_Point_With_Tracking(const vector3& coordinates, const matrix33& covariance, const cv::Mat& descriptor)
            : Point(coordinates, descriptor),
            _covariance(covariance)
        {
        }
        IMap_Point_With_Tracking::IMap_Point_With_Tracking(const vector3& coordinates, const matrix33& covariance, const cv::Mat& descriptor, const size_t id)
            : Point(coordinates, descriptor, id),
            _covariance(covariance)
        {

        }
        double IMap_Point_With_Tracking::track_point(const vector3& newPointCoordinates, const matrix33& newPointCovariance)
        {
            const matrix33& identity = matrix33::Identity(); 
            const matrix33 kalmanGain = _covariance * (_covariance + newPointCovariance).inverse();

            _coordinates = _coordinates + (kalmanGain * (newPointCoordinates - _coordinates));
            const matrix33 invGain = identity - kalmanGain;
            _covariance = (invGain * _covariance * invGain.transpose()) + (kalmanGain * newPointCovariance * kalmanGain.transpose());

            return (_coordinates - newPointCoordinates).norm();
        }


        /**
         *      Staged_Point
         */

        Staged_Point::Staged_Point(const vector3& coordinates, const matrix33& covariance, const cv::Mat& descriptor)
            : IMap_Point_With_Tracking(coordinates, covariance, descriptor)
        {
            _matchesCount = 0; 
            _lastMatchedIndex = -1;
        }
        Staged_Point::Staged_Point(const vector3& coordinates, const matrix33& covariance, const cv::Mat& descriptor, const size_t id)
            : IMap_Point_With_Tracking(coordinates, covariance, descriptor, id)
        {
            _matchesCount = 0; 
            _lastMatchedIndex = -1;
        }

        double Staged_Point::get_confidence() const 
        {
            const double confidence = static_cast<double>(_matchesCount) / static_cast<double>(Parameters::get_point_staged_age_confidence());
            return std::clamp(confidence, -1.0, 1.0);
        }

        bool Staged_Point::should_add_to_local_map() const
        {
            return (get_confidence() > Parameters::get_minimum_confidence_for_local_map());
        }


        void Staged_Point::update_unmatched(int removeNMatches)
        {
            _matchesCount -= removeNMatches;
            _screenCoordinates = cv::Point2f(-1, -1);
        }

        double Staged_Point::update_matched(const vector3& newPointCoordinates, const matrix33& covariance)
        {
            _matchesCount += 1;

            return track_point(newPointCoordinates, covariance);
        }

        bool Staged_Point::should_remove_from_staged() const
        {
            return get_confidence() <= 0; 
        }

        /**
         *     Map_Point 
         */


        Map_Point::Map_Point(const vector3& coordinates, const matrix33& covariance, const cv::Mat& descriptor) 
            : IMap_Point_With_Tracking(coordinates, covariance, descriptor)
        {
            _age = 0;
            _failTrackingCount = 0;
            _lastMatchedIndex = -1;
        }
        Map_Point::Map_Point(const vector3& coordinates, const matrix33& covariance, const cv::Mat& descriptor, const size_t id) 
            : IMap_Point_With_Tracking(coordinates, covariance, descriptor, id)
        {
            _age = 0;
            _failTrackingCount = 0;
            _lastMatchedIndex = -1;
        }

        double Map_Point::get_confidence() const
        {
            double confidence = static_cast<double>(_age) / static_cast<double>(Parameters::get_point_age_confidence());
            return std::clamp(confidence, -1.0, 1.0);
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
        void Map_Point::update_unmatched(int removeNMatches)
        {
            // decrease successful matches
            _age -= 1;
            _failTrackingCount += removeNMatches;
            _screenCoordinates = cv::Point2f(-1, -1);
        }

        /**
         * \brief Update this map point with the given informations: it is matched with another point
         */
        double Map_Point::update_matched(const vector3& newPointCoordinates, const matrix33& covariance) 
        {
            _failTrackingCount = 0;
            _age += 1;

            return track_point(newPointCoordinates, covariance);
        }



    }   /* map_management */
} /* rgbd_slam */
