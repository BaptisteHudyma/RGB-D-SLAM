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
          const cv::Mat& descriptor);

    /**
     * \brief update this point coordinates using a new detection
     * \param[in] newDetectionCoordinates The newly detected point
     * \param[in] newDetectionCovariance The newly detected point covariance
     * \return The distance between the updated position ans the previous one
     */
    double track(const utils::WorldCoordinate& newDetectionCoordinates, const matrix33& newDetectionCovariance);

  private:
    /**
     * \brief Build the caracteristics of the kalman filter
     */
    static void build_kalman_filter();

    // shared kalman filter, between all points
    inline static std::unique_ptr<tracking::SharedKalmanFilter<3, 3>> _kalmanFilter = nullptr;
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
                   const bool useAdvancedSearch = false) const override;

    bool add_to_tracked(const WorldToCameraMatrix& worldToCamera,
                        TrackedPointsObject& trackedFeatures,
                        const uint dropChance = 1000) const override;

    void draw(const WorldToCameraMatrix& worldToCamMatrix, cv::Mat& debugImage, const cv::Scalar& color) const override;

    bool is_visible(const WorldToCameraMatrix& worldToCamMatrix) const override;

    void write_to_file(std::shared_ptr<outputs::IMap_Writer> mapWriter) const override;

  protected:
    bool update_with_match(const DetectedPointType& matchedFeature,
                           const matrix33& poseCovariance,
                           const CameraToWorldMatrix& cameraToWorld) override;

    void update_no_match() override;
};

/**
 * \brief Candidate for a map point
 */
class StagedMapPoint : public MapPoint, public IStagedMapFeature<DetectedPointType>
{
  public:
    StagedMapPoint(const matrix33& poseCovariance,
                   const CameraToWorldMatrix& cameraToWorld,
                   const DetectedPointType& detectedFeature);

    bool should_remove_from_staged() const override;

    bool should_add_to_local_map() const override;

  protected:
    double get_confidence() const;
};

/**
 * \brief A map point structure, containing all the necessary informations to identify a map point in local map
 */
class LocalMapPoint : public MapPoint, public ILocalMapFeature<StagedMapPoint>
{
  public:
    LocalMapPoint(const StagedMapPoint& stagedPoint);

    bool is_lost() const override;
};

} // namespace rgbd_slam::map_management

#endif
