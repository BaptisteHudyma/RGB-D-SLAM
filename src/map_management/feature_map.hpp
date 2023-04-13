#ifndef RGBDSLAM_MAPMANAGEMENT_FEATUREMAP_HPP
#define RGBDSLAM_MAPMANAGEMENT_FEATUREMAP_HPP

#include "../outputs/logger.hpp"
#include "camera_transformation.hpp"
#include "coordinates.hpp"
#include "covariances.hpp"
#include "map_writer.hpp"
#include "parameters.hpp"
#include "pose.hpp"
#include "random.hpp"
#include "types.hpp"
#include <list>
#include <memory>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <unordered_map>

namespace rgbd_slam::map_management {

template<class DetectedFeaturesObject, class DetectedFeatureType, class FeatureMatchType, class TrackedFeaturesObject>
class IMapFeature
{
  public:
    IMapFeature() :
        _failedTrackingCount(0),
        _successivMatchedCount(0),
        _id(++_idAllocator),
        _matchIndex(FIRST_DETECTION_INDEX) {};
    IMapFeature(const size_t id) :
        _failedTrackingCount(0),
        _successivMatchedCount(0),
        _id(id),
        _matchIndex(FIRST_DETECTION_INDEX) {};

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
    virtual int find_match(const DetectedFeaturesObject& detectedFeatures,
                           const WorldToCameraMatrix& worldToCamera,
                           const vectorb& isDetectedFeatureMatched,
                           std::list<FeatureMatchType>& matches,
                           const bool shouldAddToMatches = true,
                           const bool useAdvancedSearch = false) const = 0;

    /**
     * \return True if this map feature is marked as matched
     */
    bool is_matched() const { return _matchIndex != UNMATCHED_FEATURE_INDEX; };

    /**
     * \brief mark this map feature as having no match
     */
    void mark_unmatched() { _matchIndex = UNMATCHED_FEATURE_INDEX; };

    /**
     * \brief mark this map feature as having a match at the given index
     * \param[in] matchIndex The index of the match in the detected features
     */
    void mark_matched(const int matchIndex) { _matchIndex = matchIndex; };

    /**
     * \brief Update the feature, with the corresponding match
     * \param[in] matchedFeature The detected feature matched with this map feature
     * \param[in] poseCovariance The covariance of the pose where the matchedFeature was detected
     * \param[in] cameraToWorld The matrix to transform from the camera pose where matchedFeature was detected to world
     * coordinates
     */
    void update_matched(const DetectedFeatureType& matchedFeature,
                        const matrix33& poseCovariance,
                        const CameraToWorldMatrix& cameraToWorld)
    {
        if (update_with_match(matchedFeature, poseCovariance, cameraToWorld))
        {
            // only update if tracking succedded
            _failedTrackingCount = 0;
            ++_successivMatchedCount;
        }
    };

    /**
     * \brief Update the feature, with the no match status
     */
    void update_unmatched()
    {
        mark_unmatched();

        update_no_match();
        ++_failedTrackingCount;
        _successivMatchedCount -= 1;
    };

    /**
     * \brief should write this feature to a file, using the provided mapWriter
     */
    virtual void write_to_file(std::shared_ptr<outputs::IMap_Writer> mapWriter) const = 0;

    /**
     * \brief Add the current feature to the trackedFeatures object
     * \param[in] worldToCamera A matrix to change from world to camera space
     * \param[in, out] trackedFeatures The object that contains the tracked features
     * \param[in] dropChance 1/dropChance to drop this feature and not add it
     * \return True if this was added to trackedFeatures object
     */
    virtual bool add_to_tracked(const WorldToCameraMatrix& worldToCamera,
                                TrackedFeaturesObject& trackedFeatures,
                                const uint dropChance = 1000) const = 0;

    /**
     * \brief Draw this feature on the given image
     * \param[in] worldToCamMatrix A matrix to change from world to camera space
     * \param[in, out] debugImage The image on which to display this feeature
     * \param[in] color Color of the feature to draw
     */
    virtual void draw(const WorldToCameraMatrix& worldToCamMatrix,
                      cv::Mat& debugImage,
                      const cv::Scalar& color) const = 0;

    /**
     * \brief Return true if the map feature is visible from the given position
     * \param[in] worldToCamMatrix A matrix to change from world to camera space
     * \return True if the feature should be visible
     */
    virtual bool is_visible(const WorldToCameraMatrix& worldToCamMatrix) const = 0;

    /**
     *  Members
     */
    size_t _failedTrackingCount;
    int _successivMatchedCount;
    const size_t _id; // uniq id of this point in the program
    int _matchIndex;  // index of the last matched feature id
    static const int UNMATCHED_FEATURE_INDEX = -1;
    static const int FIRST_DETECTION_INDEX = -2; // set for the first detection, to use tracking even if no match is set

  protected:
    virtual bool update_with_match(const DetectedFeatureType& matchedFeature,
                                   const matrix33& poseCovariance,
                                   const CameraToWorldMatrix& cameraToWorld) = 0;

    virtual void update_no_match() = 0;

  private:
    inline static uint _idAllocator = 0;
};

template<class DetectedFeatureType> class IStagedMapFeature
{
  public:
    virtual bool should_remove_from_staged() const = 0;
    virtual bool should_add_to_local_map() const = 0;
};

template<class StagedMapFeature> class ILocalMapFeature
{
  public:
    virtual bool is_lost() const = 0;

    cv::Vec3b _color;

    void set_color()
    {
        // set a random color for this feature
        _color[0] = utils::Random::get_random_uint(255);
        _color[1] = utils::Random::get_random_uint(255);
        _color[2] = utils::Random::get_random_uint(255);
    }
};

/**
 * Handle the storage and update of features in local and staged map
 */
template<class MapFeatureType,
         class StagedFeatureType,
         class DetectedFeaturesObject,
         class DetectedFeatureType,
         class FeatureMatchType,
         class TrackedFeaturesObject>
class Feature_Map
{
  private:
    using localMapType = std::unordered_map<size_t, MapFeatureType>;
    using stagedMapType = std::unordered_map<size_t, StagedFeatureType>;

  public:
    Feature_Map() : _isActivated(true) {}

    /**
     * \brief Reset the content of this map, empty all local maps
     */
    void reset()
    {
        if (not _isActivated)
            return;

        _localMap.clear();
        _stagedMap.clear();
    }

    void destroy(std::shared_ptr<outputs::IMap_Writer> mapWriter) const
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
     * _isDetectedFeatureMatched flags
     * \param[in] detectedFeatures The object of detected features to match
     * \param[in] worldToCamera A matrix to convert from world to camera space
     * \param[in] useAdvancedMatch If true, will restart the matching process to detected features further than if True.
     * Also less precise
     * \param[out] matches An object of matches between map object and detected features
     */
    void get_matches(const DetectedFeaturesObject& detectedFeatures,
                     const WorldToCameraMatrix& worldToCamera,
                     const bool useAdvancedMatch,
                     std::list<FeatureMatchType>& matches)
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
            // start by reseting this point
            mapFeature.mark_unmatched();

            if (not mapFeature.is_visible(worldToCamera))
                continue;

            const int matchIndex = mapFeature.find_match(
                    detectedFeatures, worldToCamera, _isDetectedFeatureMatched, matches, true, useAdvancedMatch);
            if (matchIndex >= 0)
            {
                mapFeature.mark_matched(matchIndex);
                _isDetectedFeatureMatched[matchIndex] = true;
            }
        }

        // TODO Remove this limit or make it type dependent
        // if we have enough points from local map to run the optimization, no need to add the staged points
        // Still, we need to try and match them to insure tracking and new map points
        const static uint minimumPointsForOptimization =
                Parameters::get_minimum_point_count_for_optimization() *
                3; // TODO: Why 3 ? seems about right to be sure to have enough points for the optimization process...
        const bool shouldUseStagedPoints = matches.size() < minimumPointsForOptimization;

        // search matches in staged map second
        for (auto& [mapId, mapFeature]: _stagedMap)
        {
            assert(mapId == mapFeature._id);
            // start by reseting this point
            mapFeature.mark_unmatched();

            if (not mapFeature.is_visible(worldToCamera))
                continue;

            const int matchIndex = mapFeature.find_match(detectedFeatures,
                                                         worldToCamera,
                                                         _isDetectedFeatureMatched,
                                                         matches,
                                                         shouldUseStagedPoints,
                                                         useAdvancedMatch);
            if (matchIndex >= 0)
            {
                mapFeature.mark_matched(matchIndex);
                _isDetectedFeatureMatched[matchIndex] = true;
            }
        }
    }

    /**
     * \brief Get the feature that were tracked for the last tracking step
     * \param[in] worldToCamera A matrix to convert from world to camera space
     * \param[out] trackedFeatures The object thta will contain the tracked features
     * \param[in] localMapDropChance Chance to randomly drop a local map point and not return it
     */
    void get_tracked_features(const WorldToCameraMatrix& worldToCamera,
                              TrackedFeaturesObject& trackedFeatures,
                              const uint localMapDropChance = 1000) const
    {
        if (not _isActivated)
            return;

        // local Map features
        for (const auto& [id, mapFeature]: _localMap)
        {
            assert(id == mapFeature._id);
            if (mapFeature.is_matched() and mapFeature.is_visible(worldToCamera))
            {
                mapFeature.add_to_tracked(worldToCamera, trackedFeatures, localMapDropChance);
            }
        }

        // do not track staged points, as they are not validated yet
    }

    /**
     * \brief Update this local map with a succesful tracking
     * \param[in] cameraToWorld A matrix to convert from camera to world space
     * \param[in] poseCovariance Covariance of the pose after tracking
     * \param[in] detectedFeatureObject The object containing the detected features used for the tracking
     * \param[in] mapWriter A pointer to the map writer object
     */
    void update_map(const CameraToWorldMatrix& cameraToWorld,
                    const matrix33& poseCovariance,
                    const DetectedFeaturesObject& detectedFeatureObject,
                    std::shared_ptr<outputs::IMap_Writer> mapWriter)
    {
        if (not _isActivated)
            return;
        assert(mapWriter != nullptr);

        update_local_map(cameraToWorld, poseCovariance, detectedFeatureObject, mapWriter);
        update_staged_map(cameraToWorld, poseCovariance, detectedFeatureObject);
    }

    /**
     * \brief Update this local map with a failed tracking
     * \param[in] mapWriter A pointer to the map writer object
     */
    void update_with_no_tracking(std::shared_ptr<outputs::IMap_Writer> mapWriter)
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
     * \param[in] addAllFeatures If true, will add all detected features to the staged map. If false, will only add the
     * unmatched features to staged map
     */
    void add_features_to_staged_map(const matrix33& poseCovariance,
                                    const CameraToWorldMatrix& cameraToWorld,
                                    const DetectedFeaturesObject& detectedFeatures,
                                    const bool addAllFeatures)
    {
        if (not _isActivated)
            return;

        // Add all unmatched points to staged point container
        const size_t featureVectorSize = detectedFeatures.size();
        assert(featureVectorSize == static_cast<size_t>(_isDetectedFeatureMatched.size()));
        for (unsigned int i = 0; i < featureVectorSize; ++i)
        {
            // Add all features, or add only the unmatched points
            if (addAllFeatures or not _isDetectedFeatureMatched[i])
            {
                const DetectedFeatureType& detectedfeature = detectedFeatures.at(i);
                // some features cannot be added to map
                if (detectedfeature.can_add_to_map())
                {
                    const StagedFeatureType newStagedFeature(poseCovariance, cameraToWorld, detectedfeature);
                    assert(_stagedMap.find(newStagedFeature._id) == _stagedMap.cend());
                    // add to staged map
                    _stagedMap.emplace(newStagedFeature._id, newStagedFeature);
                }
            }
        }
    }

    /**
     * \brief Mark the map feature with the given id as unmatched
     * \param[in] featureId The id of the feature to mark as unmatched
     * \return True if the feature was found, or false
     */
    bool mark_feature_with_id_as_unmatched(const size_t featureId)
    {
        if (not _isActivated)
            return false;

        if (featureId == 0)
        {
            outputs::log_error("Cannot match a feature with invalid id");
            return false;
        }
        // Check if id is in local map
        if (typename localMapType::iterator featureMapIterator = _localMap.find(featureId);
            featureMapIterator != _localMap.end())
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
        if (typename stagedMapType::iterator stagedMapIterator = _stagedMap.find(featureId);
            stagedMapIterator != _stagedMap.end())
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

        // point associated with id was not find
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
                       const bool shouldDisplayStaged = false) const
    {
        if (not _isActivated)
            return;

        if (shouldDisplayStaged)
        {
            for (const auto& [id, mapFeature]: _stagedMap)
            {
                const cv::Scalar stagedColor =
                        (mapFeature.is_matched()) ? cv::Scalar(0, 200, 255) : cv::Scalar(0, 255, 0);
                mapFeature.draw(worldToCamMatrix, debugImage, stagedColor);
            }
        }

        for (const auto& [id, mapFeature]: _localMap)
        {
            if (mapFeature.is_matched())
            {
                mapFeature.draw(worldToCamMatrix, debugImage, mapFeature._color);
            }
        }
    }

    /**
     * \brief Dectivate this local map
     */
    void deactivate() { _isActivated = false; }

    size_t get_local_map_size() const { return _localMap.size(); };
    size_t get_staged_map_size() const { return _stagedMap.size(); };
    size_t size() const { return get_local_map_size() + get_staged_map_size(); };

  protected:
    void update_local_map(const CameraToWorldMatrix& cameraToWorld,
                          const matrix33& poseCovariance,
                          const DetectedFeaturesObject& detectedFeatureObject,
                          std::shared_ptr<outputs::IMap_Writer> mapWriter)
    {
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
                // write to file
                mapFeature.write_to_file(mapWriter);

                // Remove useless point
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
        // Add correct staged points to local map
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
                // Add to local map, remove from staged points, with a copy of the id affected to the local map
                _localMap.emplace(stagedFeature._id, MapFeatureType(stagedFeature));
                assert(_localMap.at(stagedFeature._id)._id == stagedFeature._id);
                stagedFeatureIterator = _stagedMap.erase(stagedFeatureIterator);
            }
            else if (stagedFeature.should_remove_from_staged())
            {
                // Remove from staged points
                stagedFeatureIterator = _stagedMap.erase(stagedFeatureIterator);
            }
            else
            {
                // Increment
                ++stagedFeatureIterator;
            }
        }
    }

    void update_local_map_with_no_tracking(std::shared_ptr<outputs::IMap_Writer> mapWriter)
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

                // Remove useless point
                featureMapIterator = _localMap.erase(featureMapIterator);
            }
            else
            {
                ++featureMapIterator;
            }
        }
    }

    void update_staged_map_with_no_tracking()
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
                // Remove useless point
                stagedFeatureIterator = _stagedMap.erase(stagedFeatureIterator);
            }
            else
            {
                ++stagedFeatureIterator;
            }
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