#ifndef RGBDSLAM_UTILS_MATCHESCONTAINERS_HPP
#define RGBDSLAM_UTILS_MATCHESCONTAINERS_HPP

#include "types.hpp"
#include <list>
#include <memory>
#include <unordered_set>

#include "features/keypoints/keypoint_handler.hpp"
#include "features/primitives/shape_primitives.hpp"
#include "features/lines/line_detection.hpp"

namespace rgbd_slam {

enum class FeatureType
{
    Point2d,
    Point,
    Plane
};

inline std::string to_string(const FeatureType feat)
{
    switch (feat)
    {
        using enum FeatureType;
        case Point:
            return "point";
        case Point2d:
            return "point2d";
        case Plane:
            return "plane";
        default:
            return "unsupported";
    }
}

namespace map_management {

// type for matched index
typedef std::unordered_set<size_t> matchIndexSet;

/**
 * \brief Contains sets of detected features
 */
struct DetectedFeatureContainer
{
    DetectedFeatureContainer(const features::keypoints::Keypoint_Handler& newKeypointObject,
                             const features::lines::line_container& newdDetectedLines,
                             const features::primitives::plane_container& newDetectedPlanes) :
        keypointObject(newKeypointObject),
        detectedLines(newdDetectedLines),
        detectedPlanes(newDetectedPlanes),
        id(++idAllocator)
    {
    }

    const features::keypoints::Keypoint_Handler keypointObject;
    const features::lines::line_container detectedLines;
    const features::primitives::plane_container detectedPlanes;
    const size_t id; // unique id to differenciate from other detections

  private:
    inline static size_t idAllocator = 0;
};

/**
 * \brief Contains a set of features to track on the new image, before any detection
 */
struct TrackedFeaturesContainer
{
    TrackedFeaturesContainer() : trackedPoints(std::make_shared<features::keypoints::KeypointsWithIdStruct>()) {}

    std::shared_ptr<features::keypoints::KeypointsWithIdStruct> trackedPoints;
};

/**
 * \brief Base class for upgraded features
 */
struct IUpgradedFeature
{
    IUpgradedFeature(const matchIndexSet& matchIndexes) : _matchIndexes(matchIndexes) {}
    virtual ~IUpgradedFeature() = default;

    virtual FeatureType get_type() const = 0;

    const matchIndexSet _matchIndexes;
};

using UpgradedFeature_ptr = std::shared_ptr<IUpgradedFeature>;

struct UpgradedPoint2D : IUpgradedFeature
{
    UpgradedPoint2D(const WorldCoordinate& coordinates,
                    const WorldCoordinateCovariance& covariance,
                    const cv::Mat& descriptor,
                    const matchIndexSet& matchIndexes) :
        IUpgradedFeature(matchIndexes),
        _coordinates(coordinates),
        _covariance(covariance),
        _descriptor(descriptor)
    {
    }

    FeatureType get_type() const override { return FeatureType::Point; }

    WorldCoordinate _coordinates;
    WorldCoordinateCovariance _covariance;
    cv::Mat _descriptor;
};

} // namespace map_management

namespace matches_containers {

/**
 * \brief Generic feature for optimization
 */
struct IOptimizationFeature;
using feat_ptr = std::shared_ptr<IOptimizationFeature>;

struct IOptimizationFeature
{
    IOptimizationFeature(const size_t idInMap, const size_t detectedFeatureId) :
        _idInMap(idInMap),
        _detectedFeatureId(detectedFeatureId) {};

    virtual ~IOptimizationFeature() = default;

    /**
     * \brief Return the number of distance element that this feature will return
     */
    virtual size_t get_feature_part_count() const noexcept = 0;

    /**
     * \brief Return the score of this feature for an optimization.
     * This score is dependent on the minimum number of features of this type that must be used for an optimization.
     * The feature score should be 1.0 / minNumberOfFeaturesForOpti. Result should be in range ]0.0; 1.0]
     * Eg: we need at least 5 points for a 6D pose optimization, so the score for points should be 0.2
     */
    virtual double get_score() const noexcept = 0;

    /**
     * \brief Compute the distance to the matched feature, given a specific transformation matrix
     * The size of the returned vector corresponds to the part count.
     */
    virtual vectorxd get_distance(const WorldToCameraMatrix& worldToCamera) const noexcept = 0;

    /**
     * \brief Return the maximum allowed retroprojection error before rejecting this match
     */
    virtual double get_max_retroprojection_error() const noexcept = 0;

    /**
     * \brief return this feature alpha reduction (optimization weight)
     */
    virtual double get_alpha_reduction() const noexcept = 0;

    /**
     * \brief compute a random variation of this feature.
     * Should be based on the feature covariance
     */
    virtual feat_ptr compute_random_variation() const noexcept = 0;

    /**
     * \brief check the validity of the values in this structure
     */
    virtual bool is_valid() const noexcept = 0;

    /**
     * \brief return the feature type in this object
     */
    virtual FeatureType get_feature_type() const noexcept = 0;

    /// store the id of the feature in the local map/staged features
    const size_t _idInMap;

    /// Store the id of the matched detected feature
    const size_t _detectedFeatureId;
};

/**
 * \brief Store a set of inlier and a set of outliers
 */
template<class Container> struct match_sets_template
{
    Container _inliers;
    Container _outliers;

    void clear() noexcept
    {
        _inliers.clear();
        _outliers.clear();
    }

    void swap(match_sets_template& other) noexcept
    {
        _inliers.swap(other._inliers);
        _outliers.swap(other._outliers);
    }
};

using match_container = std::list<feat_ptr>;

// store a set of inliers and a set of outliers for all features
using match_sets = match_sets_template<match_container>;

} // namespace matches_containers
} // namespace rgbd_slam

#endif
