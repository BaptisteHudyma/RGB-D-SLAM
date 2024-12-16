#ifndef RGBDSLAM_MAPMANAGEMENT_LOCALMAP_HPP
#define RGBDSLAM_MAPMANAGEMENT_LOCALMAP_HPP

#include "covariances.hpp"
#include "outputs/map_writer.hpp"
#include "matches_containers.hpp"
#include "utils/pose.hpp"

#include "camera_transformation.hpp"
#include "outputs/logger.hpp"
#include "parameters.hpp"

namespace rgbd_slam::map_management {

/**
 * \brief Maintain a local (around the camera) map.
 * Handle the feature association and tracking in local space.
 * Can return matched features, and update the global map when features are estimated to be reliable.
 */
template<class... Maps> class Local_Map
{
  public:
    Local_Map()
    {
        _mapWriter = std::make_unique<outputs::OBJ_Map_Writer>("out");

        // For testing purposes, one can deactivate those maps
        /*
        foreach_map([](auto& map) {
             map.deactivate();
        });
        */
    }

    ~Local_Map()
    {
        foreach_map([this](auto& map) {
            map.destroy(_mapWriter);
        });
    }

    /**
     * \brief Return an object containing the tracked features in screen space (2D), with the associated global ids
     * \param[in] lastPose The last known pose of the observer
     */
    [[nodiscard]] TrackedFeaturesContainer get_tracked_features(const utils::Pose& lastPose) const noexcept
    {
        size_t numberOfFeaturesToTrack = 0;
        foreach_map([&numberOfFeaturesToTrack](const auto& map) {
            numberOfFeaturesToTrack += map.get_local_map_size();
        });

        // initialize output structure
        TrackedFeaturesContainer trackedFeatures;

        if (numberOfFeaturesToTrack == 0)
            return trackedFeatures;

        // TODO: check the efficiency gain of those reserve calls
        trackedFeatures.trackedPoints->reserve(numberOfFeaturesToTrack);

        const WorldToCameraMatrix& worldToCamera = utils::compute_world_to_camera_transform(
                lastPose.get_orientation_quaternion(), lastPose.get_position());

        foreach_map([&worldToCamera, &trackedFeatures](const auto& map) {
            constexpr uint refreshFrequency = parameters::detection::keypointRefreshFrequency * 2;
            map.get_tracked_features(worldToCamera, trackedFeatures, refreshFrequency);
        });

        return trackedFeatures;
    }

    /**
     * \brief Find all matches for the given detected features
     * \param[in] currentPose The pose of the observer
     * \param[in] detectedFeatures An object that contains the detected features
     */
    [[nodiscard]] matches_containers::match_container find_feature_matches(
            const utils::Pose& currentPose, const DetectedFeatureContainer& detectedFeatures) noexcept
    {
        // store the id given to the function
        _detectedFeatureId = detectedFeatures.id;

        // get transformation matrix from estimated pose
        const WorldToCameraMatrix& worldToCamera = utils::compute_world_to_camera_transform(
                currentPose.get_orientation_quaternion(), currentPose.get_position());

        matches_containers::match_container matchSets;

        // find feature matches
        const double findMatchesStartTime = static_cast<double>(cv::getTickCount());

        foreach_map([&detectedFeatures, &worldToCamera, &matchSets](auto& map) {
            map.get_matches(detectedFeatures, worldToCamera, map.minimum_features_for_opti(), matchSets);
        });

        findMatchDuration += (static_cast<double>(cv::getTickCount()) - findMatchesStartTime) / cv::getTickFrequency();

        return matchSets;
    }

    /**
     * \brief Update the local and global map. Add new features to staged and map container
     *
     * \param[in] optimizedPose The clean true pose of the observer, after optimization
     * \param[in] detectedFeatures An object that contains all the detected features
     * \param[in] outlierMatched A container for all the wrongly associated features detected in the pose
     * optimization process. They should be marked as invalid matches
     */
    void update(const utils::Pose& optimizedPose,
                const DetectedFeatureContainer& detectedFeatures,
                const matches_containers::match_container& outlierMatched)
    {
        const double updateMapStartTime = static_cast<double>(cv::getTickCount());
        assert(_detectedFeatureId == detectedFeatures.id);

        const matrix33& poseCovariance = optimizedPose.get_position_variance();
        if (not utils::is_covariance_valid(poseCovariance))
            throw std::invalid_argument("update: The given pose covariance is invalid, map wont be update");

        // Unmatch detected outliers
        mark_outliers_as_unmatched(outlierMatched);

        const CameraToWorldMatrix& cameraToWorld = utils::compute_camera_to_world_transform(
                optimizedPose.get_orientation_quaternion(), optimizedPose.get_position());

        // update all local maps
        foreach_map([this, &cameraToWorld, &poseCovariance, &detectedFeatures](auto& map) {
            map.update_map(cameraToWorld, poseCovariance, detectedFeatures, _mapWriter);
        });

        // try to upgrade to new features
        foreach_map([this, &cameraToWorld](auto& featureMap) {
            const auto& upgradedFeatures = featureMap.get_upgraded_features(cameraToWorld);

            // try to add those features to each other maps
            size_t addedFeatures = 0;
            foreach_map([&upgradedFeatures, &addedFeatures](auto& map) {
                // TODO: find a way to use this optimization
                // if (featureMap != map)
                addedFeatures += map.add_upgraded_features(upgradedFeatures);
            });

            if (addedFeatures < upgradedFeatures.size())
            {
                outputs::log_warning(
                        "Not all upgraded features could be added to the feature maps, some features have been lost");
            }
        });

        // add new features to the maps
        const bool addAllFeatures = false; // only add unmatched features
        add_features_to_map(poseCovariance, cameraToWorld, detectedFeatures, addAllFeatures);

        // add local map features to global map
        update_local_to_global();

        mapUpdateDuration += (static_cast<double>(cv::getTickCount()) - updateMapStartTime) / cv::getTickFrequency();
    }

    /**
     * \brief Update the local map when no pose could be estimated. Consider all features as unmatched
     */
    void update_no_pose() noexcept
    {
        // update the map with no tracking : matches will be reset and features can be lost
        foreach_map([this](auto& map) {
            map.update_with_no_tracking(_mapWriter);
        });
    }

    /**
     * \brief Add features to staged map
     * \param[in] poseCovariance The pose covariance of the observer, after optimization
     * \param[in] cameraToWorld The matrix to go from camera to world space
     * \param[in] detectedFeatures Contains the detected features
     * \param[in] addAllFeatures If false, will add all non matched features, if true, add all features
     * regardless of the match status
     */
    void add_features_to_map(const matrix33& poseCovariance,
                             const CameraToWorldMatrix& cameraToWorld,
                             const DetectedFeatureContainer& detectedFeatures,
                             const bool addAllFeatures)
    {
        const double addfeaturesStartTime = static_cast<double>(cv::getTickCount());

        if (not utils::is_covariance_valid(poseCovariance))
            throw std::invalid_argument("update: The given pose covariance is invalid, map wont be update");

        assert(_detectedFeatureId == detectedFeatures.id);

        // Add unmatched features to the staged map, to unsure tracking of new features
        foreach_map([&poseCovariance, &cameraToWorld, &detectedFeatures, &addAllFeatures](auto& map) {
            map.add_features_to_staged_map(poseCovariance, cameraToWorld, detectedFeatures, addAllFeatures);
        });

        mapAddFeaturesDuration +=
                (static_cast<double>(cv::getTickCount()) - addfeaturesStartTime) / cv::getTickFrequency();
    }

    /**
     * \brief Hard clean the local and staged map
     */
    void reset() noexcept
    {
        foreach_map([](auto& map) {
            map.reset();
        });
    }

    /**
     * \brief Compute a debug image to display the features
     *
     * \param[in] camPose Pose of the camera in world coordinates
     * \param[in] shouldDisplayStaged If true, will also display the content of the staged features map
     * \param[in, out] debugImage Output image
     */
    void get_debug_image(const utils::Pose& camPose, const bool shouldDisplayStaged, cv::Mat& debugImage) const noexcept
    {
        draw_image_head_band(debugImage);

        const WorldToCameraMatrix& worldToCamMatrix =
                utils::compute_world_to_camera_transform(camPose.get_orientation_quaternion(), camPose.get_position());

        // draw all map features
        foreach_map([&worldToCamMatrix, &debugImage, &shouldDisplayStaged](const auto& map) {
            map.draw_on_image(worldToCamMatrix, debugImage, shouldDisplayStaged);
        });
    }

    void show_statistics(const double meanFrameTreatmentDuration, const uint frameCount) const noexcept
    {
        static auto get_percent_of_elapsed_time = [](double treatmentTime, double totalTimeElapsed) {
            if (totalTimeElapsed <= 0)
                return 0.0;
            return (treatmentTime / totalTimeElapsed) * 100.0;
        };

        if (frameCount > 0)
        {
            const double meanMapUpdateDuration = mapUpdateDuration / static_cast<double>(frameCount);
            outputs::log(std::format("\tMean map update time is {:.4f} seconds ({:.2f}%)",
                                     meanMapUpdateDuration,
                                     get_percent_of_elapsed_time(meanMapUpdateDuration, meanFrameTreatmentDuration)));

            const double meanMapAddFeaturesDuration = mapAddFeaturesDuration / static_cast<double>(frameCount);
            outputs::log(
                    std::format("\tMean map add features time is {:.4f} seconds ({:.2f}%)",
                                meanMapAddFeaturesDuration,
                                get_percent_of_elapsed_time(meanMapAddFeaturesDuration, meanFrameTreatmentDuration)));

            const double meanFindMatchDuration = findMatchDuration / static_cast<double>(frameCount);
            outputs::log(std::format("\tMean find match time is {:.4f} seconds ({:.2f}%)",
                                     meanFindMatchDuration,
                                     get_percent_of_elapsed_time(meanFindMatchDuration, meanFrameTreatmentDuration)));
        }
    }

  protected:
    /**
     * \brief Clean the local map so it stays local, and update the global map with the good features
     */
    void update_local_to_global() noexcept
    {
        // TODO when we have a global map
    }

    /**
     * \brief draw the top information band on the debug image
     */
    void draw_image_head_band(cv::Mat& debugImage) const noexcept
    {
        assert(not debugImage.empty());

        const cv::Size& debugImageSize = debugImage.size();
        const uint imageWidth = debugImageSize.width;

        // 20 pixels
        constexpr uint bandSize_px = 20;
        const int placeInBand_px = static_cast<int>(std::floor(bandSize_px * 0.75));

        std::stringstream textFeature;
        foreach_map([&textFeature](const auto& map) {
            textFeature << "    | " << map.get_display_name() << ":" << std::format("{: >3}", map.get_staged_map_size())
                        << ":" << std::format("{: >3}", map.get_local_map_size());
        });

        constexpr double featureTextOffset = 0.15;
        const int featureLabelPosition_px = static_cast<int>(imageWidth * featureTextOffset);
        cv::putText(debugImage,
                    textFeature.str(),
                    cv::Point(featureLabelPosition_px, placeInBand_px),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(255, 255, 255, 1));
    }

    /**
     * \brief Mark all the outliers detected during optimization as unmatched
     * \param[in] outlierMatched A container of the wrong matches detected after the optimization process
     */
    void mark_outliers_as_unmatched(const matches_containers::match_container& outlierMatched) noexcept
    {
        foreach_map([&outlierMatched](auto& map) {
            map.mark_outliers_as_unmatched(outlierMatched);
        });
    }

  private:
    size_t _detectedFeatureId; // store the if of the detected feature object

    std::tuple<Maps...> _featureMaps;

    /**
     * \brief Apply a function on all map objects
     */
    template<typename F> constexpr void foreach_map(F&& function)
    {
        // transform an index to a map object
        auto paramCall = [&](const auto index) {
            std::forward<F>(function)(std::get<index>(_featureMaps));
        };

        // iterate on all the tuple indices
        auto unfold = [&]<size_t... Ints>(std::index_sequence<Ints...>) {
            (paramCall(std::integral_constant<size_t, Ints> {}), ...);
        };

        // call the unfold for each index of the sequence
        unfold(std::make_index_sequence<std::tuple_size_v<decltype(_featureMaps)>>());
    }

    /**
     * \brief Apply a function on all map objects (const version)
     */
    template<typename F> constexpr void foreach_map(F&& function) const
    {
        // transform an index to a map object
        auto paramCall = [&](const auto index) {
            std::forward<F>(function)(std::get<index>(_featureMaps));
        };

        // iterate on all the tuple indices
        auto unfold = [&]<size_t... Ints>(std::index_sequence<Ints...>) {
            (paramCall(std::integral_constant<size_t, Ints> {}), ...);
        };

        // call the unfold for each index of the sequence
        unfold(std::make_index_sequence<std::tuple_size_v<decltype(_featureMaps)>>());
    }

    std::shared_ptr<outputs::IMap_Writer> _mapWriter = nullptr;

    // Remove copy operators
    Local_Map(const Local_Map& map) = delete;
    void operator=(const Local_Map& map) = delete;

    // perf measurments
    double findMatchDuration = 0.0;
    double mapUpdateDuration = 0.0;
    double mapAddFeaturesDuration = 0.0;
};

} // namespace rgbd_slam::map_management

#endif
