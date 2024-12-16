#ifndef RGBDSLAM_MAPMANAGEMENT_FEATUREMAP_HPP
#define RGBDSLAM_MAPMANAGEMENT_FEATUREMAP_HPP

#include "covariances.hpp"
#include "outputs/map_writer.hpp"
#include "outputs/logger.hpp"

#include "matches_containers.hpp"
#include "utils/random.hpp"

#include "types.hpp"

#include <memory>

#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <stdexcept>

namespace rgbd_slam::map_management {

/**
 * This class manages the map feature ids
 */
class MapIdAllocator
{
  public:
    static constexpr size_t invalidId = 0;

    static size_t get_new_id()
    {
        _idAllocator = std::max(_idAllocator + 1, 1ul);
        return _idAllocator;
    }

  private:
    inline static size_t _idAllocator = invalidId;
};

/**
 * \brief Interface for a map feature. All map features should inherit this
 */
template<class DetectedFeaturesObject, class DetectedFeatureType, class TrackedFeaturesObject> class IMapFeature
{
  public:
    IMapFeature() : _id(MapIdAllocator::get_new_id()) {};
    explicit IMapFeature(const size_t id) : _id(id) {};

    virtual ~IMapFeature() = default;

    /**
     * \brief Searches for a match in the detectedFeatures object.
     * \param[in] detectedFeatures The object that contains the detected features for this frame
     * \param[in] worldToCamera The matrix to convert from map space to camera space. This should be an estimate of the
     * position from which detectedFeatures were observed
     * \param[in, out] isDetectedFeatureMatched A vector of booleans indicating which detected features are already
     * matched. should be updated if a match is found
     * \param[in, out] matches The object that contains the feature matches
     * \param[in] shouldAddToMatches If false, will mark feature as matched but wont add them to the match vector
     * \param[in] useAdvancedSearch If true, will search matches with a lesser accuracy, but further
     * \return the index of the match if found, or UNMATCHED_FEATURE_INDEX
     */
    [[nodiscard]] virtual int find_match(const DetectedFeaturesObject& detectedFeatures,
                                         const WorldToCameraMatrix& worldToCamera,
                                         const vectorb& isDetectedFeatureMatched,
                                         matches_containers::match_container& matches,
                                         const bool shouldAddToMatches = true,
                                         const bool useAdvancedSearch = false) const noexcept = 0;

    /**
     * \brief Return true if this feature can be upgraded to UpgradedFeature_ptr
     * \param[in] cameraToWorld The optimized pose
     * \param[out] upgradeFeature The upgraded feature, valid if this function returned true
     */
    [[nodiscard]] virtual bool compute_upgraded(const CameraToWorldMatrix& cameraToWorld,
                                                UpgradedFeature_ptr& upgradeFeature) const noexcept = 0;

    /**
     * \brief Should return true if the feature is detected as moving
     */
    [[nodiscard]] virtual bool is_moving() const noexcept = 0;

    /**
     * \return True if this map feature is marked as matched
     */
    [[nodiscard]] bool is_matched() const noexcept { return _matchIndex != UNMATCHED_FEATURE_INDEX; };

    /**
     * \brief mark this map feature as having no match
     */
    void mark_unmatched() noexcept { _matchIndex = UNMATCHED_FEATURE_INDEX; };

    /**
     * \brief mark this map feature as having a match at the given index
     * \param[in] matchIndex The index of the match in the detected features
     */
    void mark_matched(const int matchIndex) noexcept { _matchIndex = matchIndex; };

    /**
     * \brief Update the feature, with the corresponding match
     * \param[in] matchedFeature The detected feature matched with this map feature
     * \param[in] poseCovariance The covariance of the pose where the matchedFeature was detected
     * \param[in] cameraToWorld The matrix to transform from the camera pose where matchedFeature was detected to world
     * coordinates
     */
    void update_matched(const DetectedFeatureType& matchedFeature,
                        const matrix33& poseCovariance,
                        const CameraToWorldMatrix& cameraToWorld) noexcept
    {
        if (update_with_match(matchedFeature, poseCovariance, cameraToWorld))
        {
            // only update if tracking succedded
            _failedTrackingCount = 0;
            ++_successivMatchedCount;
        }
        else
        {
            // tracking failed, consider this match as a failure
            update_unmatched();
        }
    };

    /**
     * \brief Update the feature, with the no match status
     */
    void update_unmatched() noexcept
    {
        mark_unmatched();

        update_no_match();
        ++_failedTrackingCount;
        _successivMatchedCount -= 1;
    };

    /**
     * \brief should write this feature to a file, using the provided mapWriter
     */
    virtual void write_to_file(std::shared_ptr<outputs::IMap_Writer> mapWriter) const noexcept = 0;

    /**
     * \brief Add the current feature to the trackedFeatures object
     * \param[in] worldToCamera A matrix to change from world to camera space
     * \param[in, out] trackedFeatures The object that contains the tracked features
     * \param[in] dropChance 1/dropChance to drop this feature and not add it
     * \return True if this was added to trackedFeatures object
     */
    [[nodiscard]] virtual bool add_to_tracked(const WorldToCameraMatrix& worldToCamera,
                                              TrackedFeaturesObject& trackedFeatures,
                                              const uint dropChance = 1000) const noexcept = 0;

    /**
     * \brief Draw this feature on the given image
     * \param[in] worldToCamMatrix A matrix to change from world to camera space
     * \param[in, out] debugImage The image on which to display this feeature
     * \param[in] color Color of the feature to draw
     */
    virtual void draw(const WorldToCameraMatrix& worldToCamMatrix,
                      cv::Mat& debugImage,
                      const cv::Scalar& color) const noexcept = 0;

    /**
     * \brief Return true if the map feature is visible from the given position
     * \param[in] worldToCamMatrix A matrix to change from world to camera space
     * \return True if the feature should be visible
     */
    [[nodiscard]] virtual bool is_visible(const WorldToCameraMatrix& worldToCamMatrix) const noexcept = 0;

    /**
     *  Members
     */
    static constexpr int UNMATCHED_FEATURE_INDEX = -1;
    static constexpr int FIRST_DETECTION_INDEX =
            -2; // set for the first detection, to use tracking even if no match is set

    size_t _failedTrackingCount = 0;
    int _successivMatchedCount = 0;
    const size_t _id;                        // uniq id of this feature in the program (uniq id per feature type only)
    int _matchIndex = FIRST_DETECTION_INDEX; // index of the last matched feature id

  protected:
    [[nodiscard]] virtual bool update_with_match(const DetectedFeatureType& matchedFeature,
                                                 const matrix33& poseCovariance,
                                                 const CameraToWorldMatrix& cameraToWorld) noexcept = 0;

    virtual void update_no_match() noexcept = 0;
};

/**
 * \brief Interface for a staged feature. All staged features should inherit this
 */
template<class DetectedFeatureType> class IStagedMapFeature
{
  public:
    /*
    IStagedMapFeature(const matrix33& poseCovariance,
                      const CameraToWorldMatrix& cameraToWorld,
                      const DetectedFeatureType& detectedFeature) = 0;
    */

    virtual ~IStagedMapFeature() = default;

    [[nodiscard]] virtual bool should_remove_from_staged() const noexcept = 0;
    [[nodiscard]] virtual bool should_add_to_local_map() const noexcept = 0;

    [[nodiscard]] static bool can_add_to_map(const DetectedFeatureType& detectedFeature) noexcept
    {
        (void)detectedFeature;
        outputs::log_error("Called default can_add_to_map function");
        return false;
    }
};

/**
 * \brief Interface for a local map feature. All local map features should inherit this
 */
template<class StagedMapFeature> class ILocalMapFeature
{
  public:
    /*
    ILocalMapFeature(const StagedMapFeature& stagedfeature);
    */

    virtual ~ILocalMapFeature() = default;

    [[nodiscard]] virtual bool is_lost() const noexcept = 0;

    cv::Vec3b _color;

    void set_color() noexcept
    {
        // set a random color for this feature
        _color[0] = utils::Random::get_random_uint(256);
        _color[1] = utils::Random::get_random_uint(256);
        _color[2] = utils::Random::get_random_uint(256);
    }
};

/**
 * \brief Handle the storage and update of features in local and staged map
 */
template<class MapFeatureType,
         class StagedFeatureType,
         class DetectedFeaturesObject,
         class DetectedFeatureType,
         class TrackedFeaturesObject>
class Feature_Map
{
  private:
    using localMapType = std::unordered_map<size_t, MapFeatureType>;
    using stagedMapType = std::unordered_map<size_t, StagedFeatureType>;

  public:
    Feature_Map() : _isActivated(true) {}

    virtual ~Feature_Map() = default;

    /**
     * \brief Return the handled feature type
     */
    virtual FeatureType get_feature_type() const = 0;

    /**
     * \brief Return a shortened name of the feature handled (debug and display)
     */
    virtual std::string get_display_name() const = 0;

    /**
     * \brief From the given feature container return the type handled by this class
     */
    virtual DetectedFeaturesObject get_detected_feature(const DetectedFeatureContainer& features) const = 0;

    /**
     * \brief From the given tracked feature container return the type handled by this class
     * If the result is nullptr, this map cannot track features
     */
    virtual std::shared_ptr<TrackedFeaturesObject> get_tracked_features_container(
            const TrackedFeaturesContainer& tracked) const = 0;

    /**
     * \brief Return the minimum number of features for a pose optimization
     */
    virtual size_t minimum_features_for_opti() const = 0;

    /**
     * \brief Reset the content of this map, empty all local maps
     */
    void reset() noexcept
    {
        if (not _isActivated)
            return;

        _localMap.clear();
        _stagedMap.clear();
    }

    /**
     * \brief empty the map, but store map features to a map file
     */
    void destroy(std::shared_ptr<outputs::IMap_Writer> mapWriter) const noexcept
    {
        if (not _isActivated)
            return;

        // write map features to file
        for (const auto& mapFeatureIterator: _localMap)
        {
            mapFeatureIterator.second.write_to_file(mapWriter);
        }
    }

    /**
     * \brief return the object thta contains the matches between detected and map feature. Set the
     * _isDetectedFeatureMatched flags. Will relaunch the proces if not enough matches were found
     * \param[in] detectedFeatures The object of detected features to match
     * \param[in] worldToCamera A matrix to convert from world to camera space
     * \param[in] minimumFeaturesForOptimization The minimum feature count for a pose optimization
     * \param[in, out] matches An object of matches between map object and detected features
     */
    void get_matches(const DetectedFeatureContainer& detectedFeatures,
                     const WorldToCameraMatrix& worldToCamera,
                     const uint minimumFeaturesForOptimization,
                     matches_containers::match_container& matches) noexcept
    {
        const auto& detected = get_detected_feature(detectedFeatures);

        matches_containers::match_container testMatches;
        get_matches(detected, worldToCamera, false, minimumFeaturesForOptimization, testMatches);
        if (testMatches.size() < minimumFeaturesForOptimization)
        // What is the use of this metric ? TODO: document
        // or testMatches.size() < std::min(detectedFeatures.size(), get_local_map_size()) / 2)
        {
            get_matches(detected, worldToCamera, true, minimumFeaturesForOptimization, testMatches);
        }

        // merge the two
        matches.merge(testMatches);
    }

    /**
     * \brief Get the feature that were tracked for the last tracking step
     * \param[in] worldToCamera A matrix to convert from world to camera space
     * \param[out] trackedFeatures The object thta will contain the tracked features
     * \param[in] localMapDropChance Chance to randomly drop a local map feature and not return it
     */
    void get_tracked_features(const WorldToCameraMatrix& worldToCamera,
                              TrackedFeaturesContainer& trackedFeatures,
                              const uint localMapDropChance = 0) const noexcept
    {
        if (not _isActivated)
            return;

        std::shared_ptr<TrackedFeaturesObject> tracked = get_tracked_features_container(trackedFeatures);
        // this map cannot track
        if (tracked == nullptr)
            return;

        // local Map features
        for (const auto& [id, mapFeature]: _localMap)
        {
            assert(id == mapFeature._id);
            // feature was matched at the last iteration, and is visible
            if (mapFeature.is_matched() and mapFeature.is_visible(worldToCamera))
            {
                mapFeature.add_to_tracked(worldToCamera, *tracked, localMapDropChance);
            }
        }

        // do not track staged features, as they are not validated yet
    }

    /**
     * \brief Update this local map with a succesful tracking
     * \param[in] cameraToWorld A matrix to convert from camera to world space
     * \param[in] poseCovariance Covariance of the pose after tracking
     * \param[in] detectedFeatures The object containing the detected features used for the tracking
     * \param[in] mapWriter A pointer to the map writer object
     */
    void update_map(const CameraToWorldMatrix& cameraToWorld,
                    const matrix33& poseCovariance,
                    const DetectedFeatureContainer& detectedFeatures,
                    std::shared_ptr<outputs::IMap_Writer> mapWriter)
    {
        if (not _isActivated)
            return;
        if (not utils::is_covariance_valid(poseCovariance))
            throw std::invalid_argument("update_map: The given pose covariance is invalid, map wont be update");

        assert(mapWriter != nullptr);

        const auto& detected = get_detected_feature(detectedFeatures);

        update_local_map(cameraToWorld, poseCovariance, detected, mapWriter);
        update_staged_map(cameraToWorld, poseCovariance, detected);
    }

    /**
     * \brief Update this local map with a failed tracking
     * \param[in] mapWriter A pointer to the map writer object
     */
    void update_with_no_tracking(std::shared_ptr<outputs::IMap_Writer> mapWriter) noexcept
    {
        if (not _isActivated)
            return;
        assert(mapWriter != nullptr);

        update_local_map_with_no_tracking(mapWriter);
        update_staged_map_with_no_tracking();
    }

    /**
     * \brief Add a set of detected features to the staged features
     * \param[in] poseCovariance Covariance of the pose where those features were detected
     * \param[in] cameraToWorld A matrix to convert from camera to world space
     * \param[in] detectedFeatures The object that contains the detected features to add
     * \param[in] addAllFeatures If true, will add all detected features to the staged map. If false, will only add
     * the unmatched features to staged map
     */
    void add_features_to_staged_map(const matrix33& poseCovariance,
                                    const CameraToWorldMatrix& cameraToWorld,
                                    const DetectedFeatureContainer& detectedFeatures,
                                    const bool addAllFeatures)
    {
        if (not _isActivated)
            return;

        if (not utils::is_covariance_valid(poseCovariance))
            throw std::invalid_argument(
                    "add_features_to_staged_map: The given pose covariance is invalid, map wont be update");

        const auto& detected = get_detected_feature(detectedFeatures);

        // Add all unmatched features to staged feature container
        const size_t featureVectorSize = detected.size();
        assert(featureVectorSize == static_cast<size_t>(_isDetectedFeatureMatched.size()));
        for (unsigned int i = 0; i < featureVectorSize; ++i)
        {
            // Add all features, or add only the unmatched features
            if (addAllFeatures or not _isDetectedFeatureMatched[i])
            {
                const DetectedFeatureType& detectedfeature = detected.at(i);
                // some features cannot be added to map
                if (StagedFeatureType::can_add_to_map(detectedfeature))
                {
                    try
                    {
                        const StagedFeatureType newStagedFeature(poseCovariance, cameraToWorld, detectedfeature);
                        assert(_stagedMap.find(newStagedFeature._id) == _stagedMap.cend());

                        // add to staged map
                        _stagedMap.emplace(newStagedFeature._id, newStagedFeature);
                    }
                    catch (const std::exception& ex)
                    {
                        outputs::log_error(
                                get_display_name() +
                                ": Caught exception while creating the staged feature: " + std::string(ex.what()));
                    }
                }
            }
        }
    }

    /**
     * \brief Mark all outliers hanled by this map as unmatched features
     */
    void mark_outliers_as_unmatched(const matches_containers::match_container& outlierMatched) noexcept
    {
        // Mark outliers as unmatched
        for (const auto& match: outlierMatched)
        {
            if (match->get_feature_type() == get_feature_type())
            {
                const bool isOutlierRemoved = mark_feature_with_id_as_unmatched(match->_idInMap);
                // If no feature were found, this is bad. A match marked as outliers must be in the local map or
                // staged features
                if (not isOutlierRemoved)
                {
                    outputs::log_error(std::format(
                            "{}: Could not find the target feature with id {}", get_display_name(), match->_idInMap));
                }
            }
        }
    }

    /**
     * \brief Mark the map feature with the given id as unmatched
     * \param[in] featureId The id of the feature to mark as unmatched
     * \return True if the feature was found, or false
     */
    [[nodiscard]] bool mark_feature_with_id_as_unmatched(const size_t featureId) noexcept
    {
        if (not _isActivated)
            return false;

        if (featureId == 0)
        {
            outputs::log_error(get_display_name() + ": Cannot match a feature with invalid id");
            return false;
        }
        // Check if id is in local map
        typename localMapType::iterator featureMapIterator = _localMap.find(featureId);
        if (featureMapIterator != _localMap.end())
        {
            MapFeatureType& mapFeature = featureMapIterator->second;
            assert(mapFeature._id == featureId);

            if (mapFeature.is_matched())
            {
                const int matchindex = mapFeature._matchIndex;
                assert(matchindex >= 0 and matchindex < _isDetectedFeatureMatched.size());
                _isDetectedFeatureMatched[matchindex] = false;
                mapFeature.mark_unmatched();
            }
            return true;
        }

        // Check if it is in staged map
        typename stagedMapType::iterator stagedMapIterator = _stagedMap.find(featureId);
        if (stagedMapIterator != _stagedMap.end())
        {
            StagedFeatureType& mapFeature = stagedMapIterator->second;
            assert(mapFeature._id == featureId);

            if (mapFeature.is_matched())
            {
                const int matchindex = mapFeature._matchIndex;
                assert(matchindex >= 0 and matchindex < _isDetectedFeatureMatched.size());
                _isDetectedFeatureMatched[matchindex] = false;
                mapFeature.mark_unmatched();
            }
            return true;
        }

        // feature associated with id was not found
        return false;
    }

    /**
     * \brief Draw the content of this local map on the given image (draw only matched features)
     * \param[in] worldToCamMatrix A matrix to convert from world to camera space
     * \param[in, out] debugImage The image on which to draw the map content
     * \param[in] shouldDisplayStaged If true, will also display the content of the staged map
     */
    void draw_on_image(const WorldToCameraMatrix& worldToCamMatrix,
                       cv::Mat& debugImage,
                       const bool shouldDisplayStaged = false) const noexcept
    {
        if (not _isActivated)
            return;

        if (shouldDisplayStaged)
        {
            for (const auto& [id, mapFeature]: _stagedMap)
            {
                // macthed staged features are orange, unmacthed are red
                const cv::Scalar stagedColor =
                        (mapFeature.is_matched()) ? cv::Scalar(0, 200, 255) : cv::Scalar(0, 0, 255);
                mapFeature.draw(worldToCamMatrix, debugImage, stagedColor);
            }
        }

        for (const auto& [id, mapFeature]: _localMap)
        {
            mapFeature.draw(worldToCamMatrix, debugImage, mapFeature._color);
        }
    }

    /**
     * \brief Dectivate this local map
     */
    void deactivate() noexcept { _isActivated = false; }

    [[nodiscard]] size_t get_local_map_size() const noexcept { return _localMap.size(); };
    [[nodiscard]] size_t get_staged_map_size() const noexcept { return _stagedMap.size(); };
    [[nodiscard]] size_t size() const noexcept { return get_local_map_size() + get_staged_map_size(); };

    /**
     * \brief compute the upgraded features and remove them from the map
     */
    [[nodiscard]] std::vector<UpgradedFeature_ptr> get_upgraded_features(const CameraToWorldMatrix& cameraToWorld)
    {
        auto upgradedMapFeatures = get_upgraded_map_features(cameraToWorld);
        auto upgradedStagedFeatures = get_upgraded_staged_features(cameraToWorld);
        upgradedMapFeatures.insert(
                upgradedMapFeatures.end(), upgradedStagedFeatures.begin(), upgradedStagedFeatures.end());
        return upgradedMapFeatures;
    }

    /**
     * \brief Add the new features of correct type to the local map
     * \return the number of feature added to the map ( <= upgradedFeatures.size())
     */
    size_t add_upgraded_features(const std::vector<UpgradedFeature_ptr>& upgradedFeatures)
    {
        size_t addedFeatures = 0;
        for (const auto& upgraded: upgradedFeatures)
        {
            // if they are the same type, add this new feature
            if (upgraded->get_type() == get_feature_type())
            {
                add_upgraded_to_local_map(upgraded);
                addedFeatures += 1;
            }
        }
        return addedFeatures;
    }

  protected:
    // shortcut to add map features
    virtual void add_upgraded_to_local_map(const UpgradedFeature_ptr upgradedfeature) = 0;

    /**
     * \brief return the object thta contains the matches between detected and map feature. Set the
     * _isDetectedFeatureMatched flags
     * \param[in] detectedFeatures The object of detected features to match
     * \param[in] worldToCamera A matrix to convert from world to camera space
     * \param[in] useAdvancedMatch If true, will restart the matching process to detected features further than if
     * True. Also less precise
     * \param[in] minimumFeaturesForOptimization The minimum feature count for a pose optimization
     * \param[out] matches An object of matches between map object and detected features
     */
    void get_matches(const DetectedFeaturesObject& detectedFeatures,
                     const WorldToCameraMatrix& worldToCamera,
                     const bool useAdvancedMatch,
                     const uint minimumFeaturesForOptimization,
                     matches_containers::match_container& matches) noexcept
    {
        if (not _isActivated)
            return;

        // reset match status
        _isDetectedFeatureMatched = vectorb::Zero(detectedFeatures.size());
        matches.clear();

        // search matches in local map first
        for (auto& [mapId, mapFeature]: _localMap)
        {
            assert(mapId == mapFeature._id);
            // start by reseting this feature
            mapFeature.mark_unmatched();

            if (mapFeature.is_moving() or not mapFeature.is_visible(worldToCamera))
                continue;

            const int matchIndex = mapFeature.find_match(
                    detectedFeatures, worldToCamera, _isDetectedFeatureMatched, matches, true, useAdvancedMatch);
            if (matchIndex >= 0)
            {
                mapFeature.mark_matched(matchIndex);
                _isDetectedFeatureMatched[matchIndex] = true;
            }
        }

        // if we have enough features from local map to run the optimization, no need to add the staged features
        // Still, we need to try and match them to insure tracking and new map features
        // TODO: Why 3 ? seems about right to be sure to have enough features for the optimization process...
        const bool shouldUseStagedFeatures = matches.size() < minimumFeaturesForOptimization * 3;

        // search matches in staged map second
        for (auto& [mapId, mapFeature]: _stagedMap)
        {
            assert(mapId == mapFeature._id);
            // start by reseting this feature
            mapFeature.mark_unmatched();

            if (mapFeature.is_moving() or not mapFeature.is_visible(worldToCamera))
                continue;

            const int matchIndex = mapFeature.find_match(detectedFeatures,
                                                         worldToCamera,
                                                         _isDetectedFeatureMatched,
                                                         matches,
                                                         shouldUseStagedFeatures,
                                                         useAdvancedMatch);
            if (matchIndex >= 0)
            {
                mapFeature.mark_matched(matchIndex);
                _isDetectedFeatureMatched[matchIndex] = true;
            }
        }
    }

    void update_local_map(const CameraToWorldMatrix& cameraToWorld,
                          const matrix33& poseCovariance,
                          const DetectedFeaturesObject& detectedFeatureObject,
                          std::shared_ptr<outputs::IMap_Writer> mapWriter)
    {
        if (not utils::is_covariance_valid(poseCovariance))
            throw std::invalid_argument("update_local_map: The given pose covariance is invalid, map wont be update");

        typename localMapType::iterator featureMapIterator = _localMap.begin();
        while (featureMapIterator != _localMap.end())
        {
            // Update the matched/unmatched status
            MapFeatureType& mapFeature = featureMapIterator->second;
            assert(featureMapIterator->first == mapFeature._id);

            // if matched, update the features parameters
            if (mapFeature.is_matched())
            {
                assert(mapFeature._matchIndex >= 0);
                const size_t matchedFeatureIndex = mapFeature._matchIndex;
                assert(matchedFeatureIndex < detectedFeatureObject.size());

                const DetectedFeatureType& detectedFeature = detectedFeatureObject.at(matchedFeatureIndex);
                mapFeature.update_matched(detectedFeature, poseCovariance, cameraToWorld);
            }
            else
            {
                mapFeature.update_unmatched();
            }

            if (mapFeature.is_lost())
            {
                if (not mapFeature.is_moving())
                {
                    // write to file
                    mapFeature.write_to_file(mapWriter);
                }

                // Remove useless feature
                featureMapIterator = _localMap.erase(featureMapIterator);
            }
            else
            {
                ++featureMapIterator;
            }
        }
    }

    void update_staged_map(const CameraToWorldMatrix& cameraToWorld,
                           const matrix33& poseCovariance,
                           const DetectedFeaturesObject& detectedFeatureObject)
    {
        if (not utils::is_covariance_valid(poseCovariance))
            throw std::invalid_argument("update_staged_map: The given pose covariance is invalid, map wont be update");

        // Add correct staged features to local map
        typename stagedMapType::iterator stagedFeatureIterator = _stagedMap.begin();
        while (stagedFeatureIterator != _stagedMap.end())
        {
            StagedFeatureType& stagedFeature = stagedFeatureIterator->second;
            assert(stagedFeatureIterator->first == stagedFeature._id);

            // if matched, update the features parameters
            if (stagedFeature.is_matched())
            {
                assert(stagedFeature._matchIndex >= 0);
                const size_t matchedFeatureIndex = stagedFeature._matchIndex;
                assert(matchedFeatureIndex < detectedFeatureObject.size());

                const DetectedFeatureType& detectedFeature = detectedFeatureObject.at(matchedFeatureIndex);
                stagedFeature.update_matched(detectedFeature, poseCovariance, cameraToWorld);
            }
            else
            {
                stagedFeature.update_unmatched();
            }

            if (stagedFeature.should_add_to_local_map())
            {
                try
                {
                    // Add to local map, remove from staged features, with a copy of the id affected to the local map
                    _localMap.emplace(stagedFeature._id, MapFeatureType(stagedFeature));
                    assert(_localMap.at(stagedFeature._id)._id == stagedFeature._id);
                    stagedFeatureIterator = _stagedMap.erase(stagedFeatureIterator);
                }
                catch (const std::exception& ex)
                {
                    outputs::log_error(get_display_name() +
                                       ": Caught exeption while creating a map feature from a staged feature: " +
                                       std::string(ex.what()));
                }
            }
            else if (stagedFeature.should_remove_from_staged())
            {
                // Remove from staged features
                stagedFeatureIterator = _stagedMap.erase(stagedFeatureIterator);
            }
            else
            {
                // Increment
                ++stagedFeatureIterator;
            }
        }
    }

    void update_local_map_with_no_tracking(std::shared_ptr<outputs::IMap_Writer> mapWriter) noexcept
    {
        // update the local map with no matchs
        typename localMapType::iterator featureMapIterator = _localMap.begin();
        while (featureMapIterator != _localMap.end())
        {
            // Update the matched/unmatched status
            MapFeatureType& mapFeature = featureMapIterator->second;
            assert(featureMapIterator->first == mapFeature._id);

            mapFeature.update_unmatched();
            if (mapFeature.is_lost())
            {
                // write to file
                mapFeature.write_to_file(mapWriter);

                // Remove useless feature
                featureMapIterator = _localMap.erase(featureMapIterator);
            }
            else
            {
                ++featureMapIterator;
            }
        }
    }

    void update_staged_map_with_no_tracking() noexcept
    {
        // update the staged map with no matchs
        typename stagedMapType::iterator stagedFeatureIterator = _stagedMap.begin();
        while (stagedFeatureIterator != _stagedMap.end())
        {
            StagedFeatureType& stagedFeature = stagedFeatureIterator->second;
            assert(stagedFeatureIterator->first == stagedFeature._id);

            stagedFeature.update_unmatched();
            if (stagedFeature.should_remove_from_staged())
            {
                // Remove useless feature
                stagedFeatureIterator = _stagedMap.erase(stagedFeatureIterator);
            }
            else
            {
                ++stagedFeatureIterator;
            }
        }
    }

    [[nodiscard]] std::vector<UpgradedFeature_ptr> get_upgraded_map_features(
            const CameraToWorldMatrix& cameraToWorld) noexcept
    {
        std::vector<UpgradedFeature_ptr> upgradedFeatures;

        typename localMapType::iterator mapFeatureIterator = _localMap.begin();
        while (mapFeatureIterator != _localMap.end())
        {
            MapFeatureType& mapFeature = mapFeatureIterator->second;
            assert(mapFeatureIterator->first == mapFeature._id);

            UpgradedFeature_ptr upgraded;
            if (mapFeature.compute_upgraded(cameraToWorld, upgraded))
            {
                if (upgraded == nullptr)
                {
                    outputs::log_error(get_display_name() + ": compute_upgraded returned null");
                    ++mapFeatureIterator;
                    continue;
                }
                upgradedFeatures.push_back(upgraded);
                // Remove the upgraded feature
                mapFeatureIterator = _localMap.erase(mapFeatureIterator);
            }
            else
            {
                ++mapFeatureIterator;
            }
        }
        return upgradedFeatures;
    }

    [[nodiscard]] std::vector<UpgradedFeature_ptr> get_upgraded_staged_features(
            const CameraToWorldMatrix& cameraToWorld) noexcept
    {
        std::vector<UpgradedFeature_ptr> upgradedFeatures;

        typename stagedMapType::iterator stagedFeatureIterator = _stagedMap.begin();
        while (stagedFeatureIterator != _stagedMap.end())
        {
            StagedFeatureType& stagedFeature = stagedFeatureIterator->second;
            assert(stagedFeatureIterator->first == stagedFeature._id);

            UpgradedFeature_ptr upgraded;
            if (stagedFeature.compute_upgraded(cameraToWorld, upgraded))
            {
                if (upgraded == nullptr)
                {
                    outputs::log_error(get_display_name() + ": compute_upgraded returned null");
                    ++stagedFeatureIterator;
                    continue;
                }

                upgradedFeatures.push_back(upgraded);
                // Remove the upgraded feature
                stagedFeatureIterator = _stagedMap.erase(stagedFeatureIterator);
            }
            else
            {
                ++stagedFeatureIterator;
            }
        }
        return upgradedFeatures;
    }

    void add_to_local_map(const MapFeatureType& newFeature)
    {
        // check that no feature with the same id exists
        if (not _localMap.contains(newFeature._id))
        {
            _localMap.emplace(newFeature._id, newFeature);
        }
        else
        {
            outputs::log_error(get_display_name() + ": a feature with this id already exists");
        }
    }

  private:
    bool _isActivated; // if false, no updates will occur on this map object (no matches, no tracking, ...)
    localMapType _localMap;
    stagedMapType _stagedMap;
    vectorb _isDetectedFeatureMatched; // indicates if a detected feature is macthed to a local map feature
};

} // namespace rgbd_slam::map_management

#endif