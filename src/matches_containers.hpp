#ifndef RGBDSLAM_UTILS_MATCHESCONTAINERS_HPP
#define RGBDSLAM_UTILS_MATCHESCONTAINERS_HPP

#include "types.hpp"
#include <list>
#include <memory>

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
    IUpgradedFeature(const int matchIndex) : _matchIndex(matchIndex) {}
    virtual ~IUpgradedFeature() = default;

    virtual FeatureType get_type() const = 0;

    int _matchIndex;
};

using UpgradedFeature_ptr = std::shared_ptr<IUpgradedFeature>;

struct UpgradedPoint2D : IUpgradedFeature
{
    UpgradedPoint2D(const WorldCoordinate& coordinates,
                    const WorldCoordinateCovariance& covariance,
                    const cv::Mat& descriptor,
                    const int matchId) :
        IUpgradedFeature(matchId),
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
    IOptimizationFeature(const size_t idInMap) : _idInMap(idInMap) {};

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
     * \brief Return the covariance of the distance function
     */
    virtual matrixd get_distance_covariance(const WorldToCameraMatrix& worldToCamera) const noexcept = 0;

    /**
     * \brief this method is defined here only for optimization purposes.
     * The user can override it to define it's own function
     */
    virtual bool is_inlier(const WorldToCameraMatrix& worldToCamera) const
    {
        const vectorxd& std = get_distance_covariance(worldToCamera).diagonal().cwiseSqrt();
        // check that the covariance is not too huge
        // TODO: this will be truly valid when we will make all optimizations in local space (with low variances)
        if ((std.array() <= vectorxd::Constant(std.size(), 1e10).array()).all())
        {
            // get the feature distance to it's match
            const vectorxd& distances = get_distance(worldToCamera).cwiseAbs();
            // all distance should be in the computed variance 99% range
            if ((distances.array() <= 3 * std.array()).all())
            {
                return true;
            }
        }
        return false;
    }

    /**
     * \brief return this feature alpha reduction (optimization weight)
     */
    virtual double get_alpha_reduction() const noexcept = 0;

    /**
     * \brief return the feature type in this object
     */
    virtual FeatureType get_feature_type() const noexcept = 0;

    /**
     *  \brief dimention of the covariance of this feature in world space
     */
    virtual matrixd get_world_covariance() const noexcept = 0;

    /// store the id of the feature in the local map/staged features
    const size_t _idInMap;
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
