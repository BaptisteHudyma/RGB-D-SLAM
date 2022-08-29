#include "map_point.hpp"
#include "../parameters.hpp"

#include <Eigen/LU>

namespace rgbd_slam {
    namespace map_management {

        Point::Point (const vector3& coordinates, const cv::Mat& descriptor) :
            _coordinates(coordinates), 
            _descriptor(descriptor),
            _id(Point::_currentPointId)
        {
            Point::_currentPointId += 1;
        }

        Point::Point (const vector3& coordinates, const cv::Mat& descriptor, const size_t id) :
            _coordinates(coordinates), 
            _descriptor(descriptor),
            _id(id)
        {
            assert(id != INVALID_POINT_UNIQ_ID);
        }

        /**
         *     Tracked point
         */

        IMap_Point_With_Tracking::IMap_Point_With_Tracking(const vector3& coordinates, const matrix33& covariance, const cv::Mat& descriptor)
            : Point(coordinates, descriptor),
            _covariance(covariance)
        {
            _matchedScreenPoint.mark_unmatched();
        }
        IMap_Point_With_Tracking::IMap_Point_With_Tracking(const vector3& coordinates, const matrix33& covariance, const cv::Mat& descriptor, const size_t id)
            : Point(coordinates, descriptor, id),
            _covariance(covariance)
        {
            _matchedScreenPoint.mark_unmatched();
        }

        double IMap_Point_With_Tracking::track_point(const vector3& newPointCoordinates, const matrix33& newPointCovariance)
        {
            // Use a kalman filter to estimate this point position
            const matrix33& identity = matrix33::Identity(); 
            const matrix33 kalmanGain = _covariance * (_covariance + newPointCovariance).inverse();

            const vector3 newPosition = _coordinates + (kalmanGain * (newPointCoordinates - _coordinates));
            const double score = (_coordinates - newPosition).norm();

            assert(not std::isnan(newPosition.x()) and not std::isnan(newPosition.y()) and not std::isnan(newPosition.z()));

            // update this map point
            const matrix33 invGain = identity - kalmanGain;
            _covariance = (invGain * _covariance * invGain.transpose()) + (kalmanGain * newPointCovariance * kalmanGain.transpose());
            _coordinates = newPosition;

            return score; 
        }



        /**
         *      Staged_Point
         */

        Staged_Point::Staged_Point(const vector3& coordinates, const matrix33& covariance, const cv::Mat& descriptor) :
            IMap_Point_With_Tracking(coordinates, covariance, descriptor),

            _matchesCount(0)
            {
            }

        Staged_Point::Staged_Point(const vector3& coordinates, const matrix33& covariance, const cv::Mat& descriptor, const size_t id) :
            IMap_Point_With_Tracking(coordinates, covariance, descriptor, id),

            _matchesCount(0)
            {
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


        Map_Point::Map_Point(const vector3& coordinates, const matrix33& covariance, const cv::Mat& descriptor) :
            IMap_Point_With_Tracking(coordinates, covariance, descriptor),

            _failTrackingCount(0),
            _age(0)
            {
                set_random_color();
            }

        Map_Point::Map_Point(const vector3& coordinates, const matrix33& covariance, const cv::Mat& descriptor, const size_t id) :
            IMap_Point_With_Tracking(coordinates, covariance, descriptor, id),

            _failTrackingCount(0),
            _age(0)
            {
                set_random_color();
            }

        double Map_Point::get_confidence() const
        {
            double confidence = static_cast<double>(_age) / static_cast<double>(Parameters::get_point_age_confidence());
            return std::clamp(confidence, -1.0, 1.0);
        }

        void Map_Point::set_random_color()
        {
            cv::Vec3b color;
            color[0] = rand() % 255;
            color[1] = rand() % 255;
            color[2] = rand() % 255;
            _color = color;
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
