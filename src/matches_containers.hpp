#ifndef RGBDSLAM_UTILS_MATCHESCONTAINERS_HPP
#define RGBDSLAM_UTILS_MATCHESCONTAINERS_HPP

#include "types.hpp"
#include <list>
#include <memory>

namespace rgbd_slam {

enum FeatureType
{
    Point2d,
    Point,
    Plane
};

inline std::string to_string(const FeatureType feat)
{
    switch (feat)
    {
        case FeatureType::Point:
            return "point";
        case FeatureType::Point2d:
            return "point2d";
        case FeatureType::Plane:
            return "plane";
        default:
            return "unsupported";
    }
}

namespace matches_containers {

/**
 * \brief Generic feature for optimization
 */
struct IOptimizationFeature;
using feat_ptr = std::shared_ptr<IOptimizationFeature>;

struct IOptimizationFeature
{
    IOptimizationFeature(const size_t idInMap) : _idInMap(idInMap) {};
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
     * \brief return the feature type in this object
     */
    virtual FeatureType get_feature_type() const noexcept = 0;

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
