#ifndef RGBDSLAM_MAPMANAGEMENT_MAPRIMITIVE_HPP
#define RGBDSLAM_MAPMANAGEMENT_MAPRIMITIVE_HPP

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "feature_map.hpp"

#include "../parameters.hpp"
#include "../features/primitives/shape_primitives.hpp"

#include "../utils/random.hpp"
#include "../utils/coordinates.hpp"
#include "../utils/matches_containers.hpp"
#include "../utils/camera_transformation.hpp"

namespace rgbd_slam {
    namespace map_management {


        typedef features::primitives::Plane DetectedPlaneType;
        typedef features::primitives::plane_container DetectedPlaneObject;
        typedef matches_containers::PlaneMatch PlaneMatchType;
        typedef void* TrackedPlaneObject;   // TODO implement


        class Plane
        {
            public:
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

            protected:
            utils::PlaneWorldCoordinates _parametrization;  // parametrization of this plana in world space

            utils::WorldCoordinate _centroid;   // centroid of the detected plane
            cv::Mat _shapeMask; // mask of the detected plane
        };


        class MapPlane
            : public Plane, public IMapFeature<DetectedPlaneObject, DetectedPlaneType, PlaneMatchType, TrackedPlaneObject>
        {
            public:
            MapPlane() :
                IMapFeature<DetectedPlaneObject, DetectedPlaneType, PlaneMatchType, TrackedPlaneObject>()
            {
                assert(_id > 0);
            }

            MapPlane(const size_t id) :
                IMapFeature<DetectedPlaneObject, DetectedPlaneType, PlaneMatchType, TrackedPlaneObject>(id)
            {
                assert(_id > 0);
            }

            virtual int find_match(const DetectedPlaneObject& detectedFeatures, const worldToCameraMatrix& worldToCamera, const vectorb& isDetectedFeatureMatched, std::list<PlaneMatchType>& matches, const bool shouldAddToMatches = true, const bool useAdvancedSearch = false) const override 
            {
                const planeWorldToCameraMatrix& planeCameraToWorld = utils::compute_plane_world_to_camera_matrix(worldToCamera);
                // project plane in camera space
                const utils::PlaneCameraCoordinates& projectedPlane = get_parametrization().to_camera_coordinates(planeCameraToWorld);
                const utils::CameraCoordinate& planeCentroid = get_centroid().to_camera_coordinates(worldToCamera);
                const vector6& descriptor = features::primitives::Plane::compute_descriptor(projectedPlane, planeCentroid, get_contained_pixels());
                const double similarityThreshold = useAdvancedSearch ? 0.2 : 0.4;

                double smallestSimilarity = std::numeric_limits<double>::max();
                int selectedIndex = UNMATCHED_FEATURE_INDEX;

                const int detectedPlaneSize = static_cast<int>(detectedFeatures.size());
                for(int planeIndex = 0; planeIndex < detectedPlaneSize; ++planeIndex)
                {
                    if (isDetectedFeatureMatched[planeIndex])
                        // Does not allow multiple removal of a single match
                        // TODO: change this
                        continue;

                    assert(planeIndex >= 0 and planeIndex < detectedPlaneSize);
                    const features::primitives::Plane& shapePlane = detectedFeatures[planeIndex];
                    const double descriptorSimilarity  = shapePlane.get_similarity(descriptor);
                    if (descriptorSimilarity < smallestSimilarity)
                    {
                        selectedIndex = static_cast<int>(planeIndex);
                        smallestSimilarity = descriptorSimilarity;
                    }
                }

                if (selectedIndex != UNMATCHED_FEATURE_INDEX)
                {
                    if(shouldAddToMatches and smallestSimilarity < similarityThreshold)
                    {
                        const features::primitives::Plane& shapePlane = detectedFeatures[selectedIndex];
                        // TODO: replace nullptr by the plane covariance in camera space
                        matches.emplace_back(shapePlane.get_parametrization(), get_parametrization(), nullptr, _id);
                    }
                }

                return selectedIndex;
            }

            virtual bool add_to_tracked(const worldToCameraMatrix& worldToCamera, TrackedPlaneObject& trackedFeatures, const uint dropChance = 1000) const override
            {
                // silence warning for unused parameters
                (void)worldToCamera;
                (void)trackedFeatures;
                (void)dropChance;
                return false;
            }

            virtual void draw(const worldToCameraMatrix& worldToCamMatrix, cv::Mat& debugImage, const cv::Scalar& color) const override
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
            virtual bool update_with_match(const DetectedPlaneType& matchedFeature, const matrix33& poseCovariance, const cameraToWorldMatrix& cameraToWorld) override
            {
                (void)poseCovariance;
                assert(_matchIndex >= 0);

                const planeCameraToWorldMatrix& planeCameraToWorld = utils::compute_plane_camera_to_world_matrix(cameraToWorld);

                // TODO real plane tracking model
                _parametrization = matchedFeature.get_parametrization().to_world_coordinates(planeCameraToWorld);
                _centroid = matchedFeature.get_centroid().to_world_coordinates(cameraToWorld);
                _shapeMask = matchedFeature.get_shape_mask();

                return true;
            }

            virtual void update_no_match() override 
            {
            }
        };



        class StagedMapPlane
            : public virtual MapPlane, public virtual IStagedMapFeature<DetectedPlaneType>
        {
            public:
            StagedMapPlane(const matrix33& poseCovariance, const cameraToWorldMatrix& cameraToWorld, const DetectedPlaneType& detectedFeature) :
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



        class LocalMapPlane
            : public MapPlane, public ILocalMapFeature<StagedMapPlane>
        {
            public:
            LocalMapPlane(const StagedMapPlane& stagedPlane) : 
                    MapPlane(stagedPlane._id)
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

    }   // map_management
}       // rgbd_slam



#endif
