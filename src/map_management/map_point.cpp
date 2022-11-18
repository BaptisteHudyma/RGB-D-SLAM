#include "map_point.hpp"
#include "../parameters.hpp"
#include "../utils/random.hpp"

namespace rgbd_slam {
    namespace map_management {

        Point::Point (const utils::WorldCoordinate& coordinates, const cv::Mat& descriptor) :
            _coordinates(coordinates), 
            _descriptor(descriptor),
            _id(Point::_currentPointId)
        {
            Point::_currentPointId += 1;
        }

        Point::Point (const utils::WorldCoordinate& coordinates, const cv::Mat& descriptor, const size_t id) :
            _coordinates(coordinates), 
            _descriptor(descriptor),
            _id(id)
        {
            assert(id != INVALID_POINT_UNIQ_ID);
        }

        /**
         *     Tracked point
         */

        IMap_Point_With_Tracking::IMap_Point_With_Tracking(const utils::WorldCoordinate& coordinates, const matrix33& covariance, const cv::Mat& descriptor)
            : Point(coordinates, descriptor), _matchIndex(UNMATCHED_POINT_INDEX)
        {
            if (IMap_Point_With_Tracking::_kalmanFilter == nullptr)
                build_kalman_filter();

            _pointCovariance = covariance;
        }
        IMap_Point_With_Tracking::IMap_Point_With_Tracking(const utils::WorldCoordinate& coordinates, const matrix33& covariance, const cv::Mat& descriptor, const size_t id)
            : Point(coordinates, descriptor, id), _matchIndex(UNMATCHED_POINT_INDEX)
        {
            if (IMap_Point_With_Tracking::_kalmanFilter == nullptr)
                build_kalman_filter();
            
            _pointCovariance = covariance;
        }

        const double pointProcessNoise = 10;   // TODO set in parameters
        void IMap_Point_With_Tracking::build_kalman_filter()
        {
            const size_t stateDimension = 3;        //x, y, z
            const size_t measurementDimension = 3;  //x, y, z

            matrixd systemDynamics(stateDimension, stateDimension); // System dynamics matrix
            matrixd outputMatrix(measurementDimension, stateDimension); // Output matrix
            matrixd processNoiseCovariance(stateDimension, stateDimension); // Process noise covariance

            // Points are not supposed to move, so no dynamics
            systemDynamics.setIdentity();
            // we need all positions
            outputMatrix.setIdentity();

            processNoiseCovariance.setIdentity();
            processNoiseCovariance *= pointProcessNoise;

            IMap_Point_With_Tracking::_kalmanFilter = new tracking::SharedKalmanFilter(systemDynamics, outputMatrix, processNoiseCovariance);
        }

        double IMap_Point_With_Tracking::track_point(const utils::WorldCoordinate& newPointCoordinates, const matrix33& newPointCovariance)
        {
            assert(IMap_Point_With_Tracking::_kalmanFilter != nullptr);

            const std::pair<vector3, matrix33>& res = IMap_Point_With_Tracking::_kalmanFilter->get_new_state(_coordinates, _pointCovariance, newPointCoordinates, newPointCovariance);

            const double score = (_coordinates - res.first).norm();
            
            _coordinates = res.first;
            _pointCovariance = res.second;
            return score;
        }



        /**
         *      Staged_Point
         */

        Staged_Point::Staged_Point(const utils::WorldCoordinate& coordinates, const matrix33& covariance, const cv::Mat& descriptor) :
            IMap_Point_With_Tracking(coordinates, covariance, descriptor),

            _matchesCount(0)
            {
            }

        Staged_Point::Staged_Point(const utils::WorldCoordinate& coordinates, const matrix33& covariance, const cv::Mat& descriptor, const size_t id) :
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

        double Staged_Point::update_matched(const utils::WorldCoordinate& newPointCoordinates, const matrix33& covariance)
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


        Map_Point::Map_Point(const utils::WorldCoordinate& coordinates, const matrix33& covariance, const cv::Mat& descriptor) :
            IMap_Point_With_Tracking(coordinates, covariance, descriptor),

            _failTrackingCount(0),
            _age(0)
            {
                set_random_color();
            }

        Map_Point::Map_Point(const utils::WorldCoordinate& coordinates, const matrix33& covariance, const cv::Mat& descriptor, const size_t id) :
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
            color[0] = utils::Random::get_random_uint(255);
            color[1] = utils::Random::get_random_uint(255);
            color[2] = utils::Random::get_random_uint(255);
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
        double Map_Point::update_matched(const utils::WorldCoordinate& newPointCoordinates, const matrix33& covariance) 
        {
            _failTrackingCount = 0;
            _age += 1;

            return track_point(newPointCoordinates, covariance);
        }



    }   /* map_management */
} /* rgbd_slam */
