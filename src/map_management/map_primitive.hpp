#ifndef RGBDSLAM_MAPMANAGEMENT_MAPRIMITIVE_HPP
#define RGBDSLAM_MAPMANAGEMENT_MAPRIMITIVE_HPP

#include "../features/primitives/shape_primitives.hpp"
#include "../parameters.hpp"
#include "../tracking/kalman_filter.hpp"
#include "../utils/camera_transformation.hpp"
#include "../utils/coordinates.hpp"
#include "../utils/matches_containers.hpp"
#include "../utils/random.hpp"
#include "feature_map.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

namespace rgbd_slam {
namespace map_management {

typedef features::primitives::Plane DetectedPlaneType;
typedef features::primitives::plane_container DetectedPlaneObject;
typedef matches_containers::PlaneMatch PlaneMatchType;
typedef void* TrackedPlaneObject; // TODO implement

class Plane
{
  public:
    Plane()
    {
        _parametrization.setZero();
        _covariance.setIdentity();
        _centroid.setZero();

        build_kalman_filter();
    }

    /**
     * \brief Return the number of pixels in this plane mask
     */
    uint get_contained_pixels() const
    {
        const static uint cellSize = Parameters::get_depth_map_patch_size();
        const static uint pixelPerCell = cellSize * cellSize;
        return cv::countNonZero(_shapeMask) * pixelPerCell;
    }

    utils::PlaneWorldCoordinates get_parametrization() const
    {
        return _parametrization;
    }
    utils::WorldCoordinate get_centroid() const
    {
        return _centroid;
    }
    cv::Mat get_mask() const
    {
        return _shapeMask;
    }

    /**
     * \brief Update this plane coordinates using a new detection
     *
     * \param[in] newDetectionParameters The detected plane parameters
     * \param[in] newCovariance The covariance of the newly detected feature
     *
     * \return The update score (distance between old and new parametrization)
     */
    double track(const utils::PlaneWorldCoordinates& newDetectionParameters,
                 const matrix44& newDetectionCovariance,
                 const utils::WorldCoordinate& detectedCentroid)
    {
        assert(_kalmanFilter != nullptr);

        const std::pair<vector4, matrix44>& res = _kalmanFilter->get_new_state(
                _parametrization, _covariance, newDetectionParameters, newDetectionCovariance);
        const vector4& newEstimatedParameters = res.first;
        const matrix44& newEstimatedCovariance = res.second;

        const double score = (_parametrization - res.first).norm();

        // source: Revisiting Uncertainty Analysis for Optimum Planes Extracted from 3D Range Sensor Point-Clouds

        // covariance update
        Eigen::SelfAdjointEigenSolver<matrix44> covarianceSolver(newEstimatedCovariance);
        const double smallestEigenValue = covarianceSolver.eigenvalues()(0);
        const vector4& smallestEigenVector = covarianceSolver.eigenvectors().col(0).normalized();
        _covariance =
                newEstimatedCovariance - smallestEigenValue * smallestEigenVector * smallestEigenVector.transpose();

        // parameters update
        vector4 renormalizedVector = newEstimatedParameters / sqrt(pow(newEstimatedParameters.head(3).norm(), 2.0) +
                                                                   pow(newEstimatedParameters(3), 2.0));
        renormalizedVector /= sqrt(1.0 - pow(renormalizedVector(3), 2.0));
        _parametrization << renormalizedVector.head(3).normalized(), renormalizedVector(3);

        // update centroid
        _centroid = detectedCentroid;

        // static sanity checks
        assert(_covariance.diagonal()(0) >= 0 and _covariance.diagonal()(1) >= 0 and _covariance.diagonal()(2) >= 0 and
               _covariance.diagonal()(3) >= 0);
        assert(not std::isnan(_parametrization.x()) and not std::isnan(_parametrization.y()) and
               not std::isnan(_parametrization.z()) and not std::isnan(_parametrization.w()));
        return score;
    }

  protected:
    utils::PlaneWorldCoordinates _parametrization; // parametrization of this plane in world space
    matrix44 _covariance;                          // covariance of this plane in world space
    utils::WorldCoordinate _centroid;              // centroid of the detected plane
    cv::Mat _shapeMask;                            // mask of the detected plane

  private:
    /**
     * \brief Build the parameter kalman filter
     */
    static void build_kalman_filter()
    {
        if (_kalmanFilter == nullptr)
        {
            const double parametersProcessNoise = 0; // TODO set in parameters
            const size_t stateDimension = 4;         // nx, ny, nz, d
            const size_t measurementDimension = 4;   // nx, ny, nz, d

            matrixd systemDynamics(stateDimension, stateDimension);         // System dynamics matrix
            matrixd outputMatrix(measurementDimension, stateDimension);     // Output matrix
            matrixd processNoiseCovariance(stateDimension, stateDimension); // Process noise covariance

            // Points are not supposed to move, so no dynamics
            systemDynamics.setIdentity();
            // we need all positions
            outputMatrix.setIdentity();

            processNoiseCovariance.setIdentity();
            processNoiseCovariance *= parametersProcessNoise;

            _kalmanFilter = new tracking::SharedKalmanFilter(systemDynamics, outputMatrix, processNoiseCovariance);
        }
    }
    // shared kalman filter, between all planes
    inline static tracking::SharedKalmanFilter* _kalmanFilter = nullptr;
};

class MapPlane :
    public Plane,
    public IMapFeature<DetectedPlaneObject, DetectedPlaneType, PlaneMatchType, TrackedPlaneObject>
{
  public:
    MapPlane() : IMapFeature<DetectedPlaneObject, DetectedPlaneType, PlaneMatchType, TrackedPlaneObject>()
    {
        assert(_id > 0);
    }

    MapPlane(const size_t id) :
        IMapFeature<DetectedPlaneObject, DetectedPlaneType, PlaneMatchType, TrackedPlaneObject>(id)
    {
        assert(_id > 0);
    }

    virtual int find_match(const DetectedPlaneObject& detectedFeatures,
                           const worldToCameraMatrix& worldToCamera,
                           const vectorb& isDetectedFeatureMatched,
                           std::list<PlaneMatchType>& matches,
                           const bool shouldAddToMatches = true,
                           const bool useAdvancedSearch = false) const override
    {
        const planeWorldToCameraMatrix& planeCameraToWorld = utils::compute_plane_world_to_camera_matrix(worldToCamera);
        // project plane in camera space
        const utils::PlaneCameraCoordinates& projectedPlane =
                get_parametrization().to_camera_coordinates(planeCameraToWorld);
        const utils::CameraCoordinate& planeCentroid = get_centroid().to_camera_coordinates(worldToCamera);
        const vector6& descriptor =
                features::primitives::Plane::compute_descriptor(projectedPlane, planeCentroid, get_contained_pixels());
        const double similarityThreshold = useAdvancedSearch ? 0.2 : 0.4;

        double smallestSimilarity = std::numeric_limits<double>::max();
        int selectedIndex = UNMATCHED_FEATURE_INDEX;

        // search best match score
        const int detectedPlaneSize = static_cast<int>(detectedFeatures.size());
        for (int planeIndex = 0; planeIndex < detectedPlaneSize; ++planeIndex)
        {
            if (isDetectedFeatureMatched[planeIndex])
                // Does not allow multiple removal of a single match
                // TODO: change this
                continue;

            assert(planeIndex >= 0 and planeIndex < detectedPlaneSize);
            const features::primitives::Plane& shapePlane = detectedFeatures[planeIndex];

            // compute a similarity score
            const double descriptorSimilarity = shapePlane.get_similarity(descriptor);
            if (descriptorSimilarity < smallestSimilarity)
            {
                selectedIndex = static_cast<int>(planeIndex);
                smallestSimilarity = descriptorSimilarity;
            }
        }

        if (selectedIndex == UNMATCHED_FEATURE_INDEX or smallestSimilarity >= similarityThreshold)
            return UNMATCHED_FEATURE_INDEX;

        if (shouldAddToMatches)
        {
            const features::primitives::Plane& shapePlane = detectedFeatures[selectedIndex];
            // TODO: replace nullptr by the plane covariance in camera space
            matches.emplace_back(shapePlane.get_parametrization(), get_parametrization(), nullptr, nullptr, _id);
        }

        return selectedIndex;
    }

    virtual bool add_to_tracked(const worldToCameraMatrix& worldToCamera,
                                TrackedPlaneObject& trackedFeatures,
                                const uint dropChance = 1000) const override
    {
        // silence warning for unused parameters
        (void)worldToCamera;
        (void)trackedFeatures;
        (void)dropChance;
        return false;
    }

    virtual void draw(const worldToCameraMatrix& worldToCamMatrix,
                      cv::Mat& debugImage,
                      const cv::Scalar& color) const override
    {
        // silence unused parameter warning
        (void)worldToCamMatrix;

        assert(not get_mask().empty());

        const double maskAlpha = 0.3;
        const cv::Size& debugImageSize = debugImage.size();

        cv::Mat planeMask, planeColorMask;
        // Resize with no interpolation
        cv::resize(get_mask() * 255, planeMask, debugImageSize, 0, 0, cv::INTER_NEAREST);
        cv::cvtColor(planeMask, planeColorMask, cv::COLOR_GRAY2BGR);
        assert(planeMask.size == debugImage.size);
        assert(planeMask.size == debugImage.size);
        assert(planeColorMask.type() == debugImage.type());

        // merge with debug image
        planeColorMask.setTo(color, planeMask);
        cv::Mat maskedInput, ImaskedInput;
        // get masked original image, with the only visible part being the plane part
        debugImage.copyTo(maskedInput, planeMask);
        // get masked original image, with the only visible part being the non plane part
        debugImage.copyTo(ImaskedInput, 255 - planeMask);

        cv::addWeighted(maskedInput, (1 - maskAlpha), planeColorMask, maskAlpha, 0.0, maskedInput);

        debugImage = maskedInput + ImaskedInput;
    }

    virtual bool is_visible(const worldToCameraMatrix& worldToCamMatrix) const override
    {
        // TODO
        (void)worldToCamMatrix;
        return true;
    }

  protected:
    virtual bool update_with_match(const DetectedPlaneType& matchedFeature,
                                   const matrix33& poseCovariance,
                                   const cameraToWorldMatrix& cameraToWorld) override
    {
        assert(_matchIndex >= 0);

        const planeCameraToWorldMatrix& planeCameraToWorld = utils::compute_plane_camera_to_world_matrix(cameraToWorld);

        matrix44 detectedPlaneCovariance;
        detectedPlaneCovariance.setIdentity();
        track(matchedFeature.get_parametrization().to_world_coordinates(planeCameraToWorld),
              detectedPlaneCovariance, // TODO: transform covariance to world
              matchedFeature.get_centroid().to_world_coordinates(cameraToWorld));

        _shapeMask = matchedFeature.get_shape_mask();

        return true;
    }

    virtual void update_no_match() override
    {
    }
};

class StagedMapPlane : public virtual MapPlane, public virtual IStagedMapFeature<DetectedPlaneType>
{
  public:
    StagedMapPlane(const matrix33& poseCovariance,
                   const cameraToWorldMatrix& cameraToWorld,
                   const DetectedPlaneType& detectedFeature) :
        MapPlane()
    {
        (void)poseCovariance;

        const planeCameraToWorldMatrix& planeCameraToWorld = utils::compute_plane_camera_to_world_matrix(cameraToWorld);
        _parametrization = detectedFeature.get_parametrization().to_world_coordinates(planeCameraToWorld),
        _centroid = detectedFeature.get_centroid().to_world_coordinates(cameraToWorld),
        _shapeMask = detectedFeature.get_shape_mask();
    }

    virtual bool should_remove_from_staged() const override
    {
        return _failedTrackingCount >= 2;
    }

    virtual bool should_add_to_local_map() const override
    {
        return _successivMatchedCount >= 1;
    }
};

class LocalMapPlane : public MapPlane, public ILocalMapFeature<StagedMapPlane>
{
  public:
    LocalMapPlane(const StagedMapPlane& stagedPlane) : MapPlane(stagedPlane._id)
    {
        // new map point, new color
        set_color();

        _matchIndex = stagedPlane._matchIndex;
        _successivMatchedCount = stagedPlane._successivMatchedCount;

        _parametrization = stagedPlane.get_parametrization();
        _centroid = stagedPlane.get_centroid();
        _shapeMask = stagedPlane.get_mask();
    }

    virtual bool is_lost() const override
    {
        const static size_t maximumUnmatchBeforeremoval = Parameters::get_maximum_unmatched_before_removal();
        return _failedTrackingCount >= maximumUnmatchBeforeremoval;
    }
};

} // namespace map_management
} // namespace rgbd_slam

#endif
