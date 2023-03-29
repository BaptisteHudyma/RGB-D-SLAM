#ifndef RGBDSLAM_MAPMANAGEMENT_MAPPOINT_HPP
#define RGBDSLAM_MAPMANAGEMENT_MAPPOINT_HPP

#include "../features/keypoints/keypoint_handler.hpp"
#include "../tracking/kalman_filter.hpp"
#include "../utils/coordinates.hpp"
#include "feature_map.hpp"
#include "matches_containers.hpp"
#include "parameters.hpp"
#include "types.hpp"
#include <memory>
#include <opencv2/opencv.hpp>

namespace rgbd_slam::map_management {

const size_t INVALID_POINT_UNIQ_ID = 0; // This id indicates an invalid unique id for a map point

struct Point
{
    // world coordinates
    utils::WorldCoordinate _coordinates;
    // 3D descriptor (ORB)
    cv::Mat _descriptor;
    // position covariance
    WorldCoordinateCovariance _covariance;

    Point(const utils::WorldCoordinate& coordinates,
          const WorldCoordinateCovariance& covariance,
          const cv::Mat& descriptor) :
        _coordinates(coordinates),
        _descriptor(descriptor),
        _covariance(covariance)
    {
        build_kalman_filter();

        assert(not _descriptor.empty() and _descriptor.cols > 0);
        assert(not std::isnan(_coordinates.x()) and not std::isnan(_coordinates.y()) and
               not std::isnan(_coordinates.z()));
    };

    /**
     * \brief update this point coordinates using a new detection
     *
     * \param[in] newDetectionCoordinates The newly detected point
     * \param[in] newDetectionCovariance The newly detected point covariance
     *
     * \return The distance between the updated position ans the previous one
     */
    double track(const utils::WorldCoordinate& newDetectionCoordinates, const matrix33& newDetectionCovariance)
    {
        assert(_kalmanFilter != nullptr);

        const std::pair<vector3, matrix33>& res = _kalmanFilter->get_new_state(
                _coordinates, _covariance.base(), newDetectionCoordinates, newDetectionCovariance);
        const vector3& newCoordinates = res.first;
        const matrix33& newCovariance = res.second;

        const double score = (_coordinates - newCoordinates).norm();

        _coordinates = newCoordinates.base();
        _covariance.base() = newCovariance;
        assert(not std::isnan(_coordinates.x()) and not std::isnan(_coordinates.y()) and
               not std::isnan(_coordinates.z()));
        return score;
    }

  private:
    /**
     * \brief Build the caracteristics of the kalman filter
     */
    static void build_kalman_filter()
    {
        if (_kalmanFilter == nullptr)
        {
            // gain 10mm of uncertainty at each iteration
            const double pointProcessNoise = 0;    // TODO set in parameters
            const size_t stateDimension = 3;       // x, y, z
            const size_t measurementDimension = 3; // x, y, z

            matrixd systemDynamics(stateDimension, stateDimension);         // System dynamics matrix
            matrixd outputMatrix(measurementDimension, stateDimension);     // Output matrix
            matrixd processNoiseCovariance(stateDimension, stateDimension); // Process noise covariance

            // Points are not supposed to move, so no dynamics
            systemDynamics.setIdentity();
            // we need all positions
            outputMatrix.setIdentity();

            processNoiseCovariance.setIdentity();
            processNoiseCovariance *= pointProcessNoise;

            _kalmanFilter = std::make_unique<tracking::SharedKalmanFilter>(
                    systemDynamics, outputMatrix, processNoiseCovariance);
        }
    }
    // shared kalman filter, between all points
    inline static std::unique_ptr<tracking::SharedKalmanFilter> _kalmanFilter = nullptr;
};

using DetectedKeypointsObject = features::keypoints::Keypoint_Handler;
using DetectedPointType = features::keypoints::DetectedKeyPoint;
using PointMatchType = matches_containers::PointMatch;
using TrackedPointsObject = features::keypoints::KeypointsWithIdStruct;

class MapPoint :
    public Point,
    public IMapFeature<DetectedKeypointsObject, DetectedPointType, PointMatchType, TrackedPointsObject>
{
  public:
    MapPoint(const utils::WorldCoordinate& coordinates,
             const WorldCoordinateCovariance& covariance,
             const cv::Mat& descriptor) :
        Point(coordinates, covariance, descriptor),
        IMapFeature<DetectedKeypointsObject, DetectedPointType, PointMatchType, TrackedPointsObject>()
    {
        assert(_id > 0);
    }

    MapPoint(const utils::WorldCoordinate& coordinates,
             const WorldCoordinateCovariance& covariance,
             const cv::Mat& descriptor,
             const size_t id) :
        Point(coordinates, covariance, descriptor),
        IMapFeature<DetectedKeypointsObject, DetectedPointType, PointMatchType, TrackedPointsObject>(id)
    {
        assert(_id > 0);
    }

    int find_match(const DetectedKeypointsObject& detectedFeatures,
                   const WorldToCameraMatrix& worldToCamera,
                   const vectorb& isDetectedFeatureMatched,
                   std::list<PointMatchType>& matches,
                   const bool shouldAddToMatches = true,
                   const bool useAdvancedSearch = false) const override
    {
        static const double searchSpaceRadius = Parameters::get_search_matches_distance();
        static const double advancedSearchSpaceRadius = Parameters::get_search_matches_distance() * 2;
        const double searchRadius = useAdvancedSearch ? advancedSearchSpaceRadius : searchSpaceRadius;

        // try to match with tracking
        const int invalidfeatureIndex = features::keypoints::INVALID_MATCH_INDEX;
        int matchIndex = detectedFeatures.get_tracking_match_index(_id, isDetectedFeatureMatched);
        if (matchIndex == invalidfeatureIndex)
        {
            // No match: try to find match in a window around the point
            utils::ScreenCoordinate2D projectedMapPoint;
            const bool isScreenCoordinatesValid = _coordinates.to_screen_coordinates(worldToCamera, projectedMapPoint);
            if (isScreenCoordinatesValid)
            {
                matchIndex = detectedFeatures.get_match_index(
                        projectedMapPoint, _descriptor, isDetectedFeatureMatched, searchRadius);
            }
        }

        if (matchIndex == invalidfeatureIndex)
        {
            // unmatched point
            return UNMATCHED_FEATURE_INDEX;
        }

        assert(matchIndex >= 0);
        assert(static_cast<Eigen::Index>(matchIndex) < isDetectedFeatureMatched.size());
        if (isDetectedFeatureMatched[matchIndex])
        {
            // point was already matched
            outputs::log_error("The requested point unique index is already matched");
        }

        if (shouldAddToMatches)
        {
            matches.emplace_back(detectedFeatures.get_keypoint(matchIndex), _coordinates, _covariance.diagonal(), _id);
        }
        return matchIndex;
    }

    bool add_to_tracked(const WorldToCameraMatrix& worldToCamera,
                        TrackedPointsObject& trackedFeatures,
                        const uint dropChance = 1000) const override
    {
        const bool shouldNotDropPoint = (dropChance == 0) or (utils::Random::get_random_uint(dropChance) != 0);

        assert(not std::isnan(_coordinates.x()) and not std::isnan(_coordinates.y()) and
               not std::isnan(_coordinates.z()));
        if (shouldNotDropPoint)
        {
            utils::ScreenCoordinate2D screenCoordinates;
            if (_coordinates.to_screen_coordinates(worldToCamera, screenCoordinates))
            {
                // use previously known screen coordinates
                trackedFeatures.add(_id, screenCoordinates.x(), screenCoordinates.y());

                return true;
            }
        }
        // point was not added
        return false;
    }

    void draw(const WorldToCameraMatrix& worldToCamMatrix, cv::Mat& debugImage, const cv::Scalar& color) const override
    {
        utils::ScreenCoordinate2D screenPoint;
        const bool isCoordinatesValid = _coordinates.to_screen_coordinates(worldToCamMatrix, screenPoint);

        if (isCoordinatesValid)
        {
            cv::circle(debugImage,
                       cv::Point(static_cast<int>(screenPoint.x()), static_cast<int>(screenPoint.y())),
                       3,
                       color,
                       -1);
        }
    }

    bool is_visible(const WorldToCameraMatrix& worldToCamMatrix) const override
    {
        static const uint screenSizeX = Parameters::get_camera_1_size_x();
        static const uint screenSizeY = Parameters::get_camera_1_size_y();

        if (utils::ScreenCoordinate projectedScreenCoordinates;
            _coordinates.to_screen_coordinates(worldToCamMatrix, projectedScreenCoordinates))
        {
            return
                    // in screen space
                    projectedScreenCoordinates.x() >= 0 and projectedScreenCoordinates.x() <= screenSizeX and
                    projectedScreenCoordinates.y() >= 0 and projectedScreenCoordinates.y() <= screenSizeY and
                    // in front of the camera
                    projectedScreenCoordinates.z() >= 0;
        }
        return false;
    }

  protected:
    bool update_with_match(const DetectedPointType& matchedFeature,
                           const matrix33& poseCovariance,
                           const CameraToWorldMatrix& cameraToWorld) override
    {
        assert(_matchIndex >= 0);

        if (const utils::ScreenCoordinate& matchedScreenPoint = matchedFeature._coordinates;
            utils::is_depth_valid(matchedScreenPoint.z()))
        {
            // transform screen point to world point
            const utils::WorldCoordinate& worldPointCoordinates =
                    matchedScreenPoint.to_world_coordinates(cameraToWorld);
            // get a measure of the estimated variance of the new world point
            const CameraCoordinateCovariance& cameraPointCovariance =
                    utils::get_camera_point_covariance(matchedScreenPoint);
            const matrix33& worldCovariance = poseCovariance + cameraPointCovariance.base();
            // update this map point errors & position
            track(worldPointCoordinates, worldCovariance);

            // If a new descriptor is available, update it
            if (const cv::Mat& descriptor = matchedFeature._descriptor; not descriptor.empty())
                _descriptor = descriptor;

            return true;
        }
        else
        {
            // TODO: Point is 2D, handle separatly
        }
        return false;
    }

    void update_no_match() override
    {
        // do nothing
    }
};

/**
 * \brief Candidate for a map point
 */
class StagedMapPoint : public MapPoint, public IStagedMapFeature<DetectedPointType>
{
  public:
    StagedMapPoint(const matrix33& poseCovariance,
                   const CameraToWorldMatrix& cameraToWorld,
                   const DetectedPointType& detectedFeature) :
        MapPoint(detectedFeature._coordinates.to_world_coordinates(cameraToWorld),
                 utils::get_world_point_covariance(detectedFeature._coordinates, poseCovariance),
                 detectedFeature._descriptor)
    {
    }

    bool should_remove_from_staged() const override { return get_confidence() <= 0; }

    bool should_add_to_local_map() const override
    {
        const static double minimumConfidenceForLocalMap = Parameters::get_minimum_confidence_for_local_map();
        return (get_confidence() > minimumConfidenceForLocalMap);
    }

  protected:
    double get_confidence() const
    {
        const static double stagedPointconfidence = static_cast<double>(Parameters::get_point_staged_age_confidence());
        const double confidence = static_cast<double>(_successivMatchedCount) / stagedPointconfidence;
        return std::clamp(confidence, -1.0, 1.0);
    }
};

/**
 * \brief A map point structure, containing all the necessary informations to identify a map point in local map
 */
class LocalMapPoint : public MapPoint, public ILocalMapFeature<StagedMapPoint>
{
  public:
    LocalMapPoint(const StagedMapPoint& stagedPoint) :
        MapPoint(stagedPoint._coordinates, stagedPoint._covariance, stagedPoint._descriptor, stagedPoint._id)
    {
        // new map point, new color
        set_color();

        _matchIndex = stagedPoint._matchIndex;
        _successivMatchedCount = stagedPoint._successivMatchedCount;
    }

    bool is_lost() const override
    {
        const static uint maximumUnmatchBeforeRemoval = Parameters::get_maximum_unmatched_before_removal();
        return (_failedTrackingCount > maximumUnmatchBeforeRemoval);
    }
};

} // namespace rgbd_slam::map_management

#endif
