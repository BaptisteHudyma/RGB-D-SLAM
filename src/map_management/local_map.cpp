#include "local_map.hpp"

#include "../parameters.hpp"
#include "../tracking/triangulation.hpp"
#include "../utils/camera_transformation.hpp"
#include "../utils/covariances.hpp"
#include "../utils/coordinates.hpp"
#include "../utils/random.hpp"
#include "../outputs/logger.hpp"
#include "map_point.hpp"
#include "map_primitive.hpp"
#include "matches_containers.hpp"
#include "types.hpp"

namespace rgbd_slam {
    namespace map_management {


        /**
         * LOCAL UTILS FUNCTIONS
         */

        /**
         * \brief My add a point to the tracked feature object, used to add optical flow tracking
         *
         * \param[in] mapPoint The map point, to add to the tracked features depending on some conditions
         * \param[in, out] keypointsWithIds The association structure for keypoints and their uniq ids
         * \param[in] dropChance 1/dropChance that this point can be randomly dropped, and will not be added to the keypointsWithIds object
         */
        void add_point_to_tracked_features(const worldToCameraMatrix& worldToCamera, const IMap_Point_With_Tracking& mapPoint, features::keypoints::KeypointsWithIdStruct& keypointsWithIds, const uint dropChance = 1000)
        {
            const bool shouldNotDropPoint = utils::Random::get_random_uint(dropChance) != 0;

            const utils::WorldCoordinate& coordinates = mapPoint._coordinates;
            assert(not std::isnan(coordinates.x()) and not std::isnan(coordinates.y()) and not std::isnan(coordinates.z()));
            if (shouldNotDropPoint and mapPoint.is_matched())
            {
                utils::ScreenCoordinate2D screenCoordinates;
                if (mapPoint._coordinates.to_screen_coordinates(worldToCamera, screenCoordinates))
                {
                    // use previously known screen coordinates
                    keypointsWithIds._keypoints.push_back(
                        cv::Point2f(
                                    static_cast<float>(screenCoordinates.x()),
                                    static_cast<float>(screenCoordinates.y())
                                    )
                            );
                    keypointsWithIds._ids.push_back(mapPoint._id);
                }
            }
        }

        /**
         * PUBLIC
         */

        Local_Map::Local_Map()
        {
            // Check constants
            assert(features::keypoints::INVALID_MAP_POINT_ID == INVALID_POINT_UNIQ_ID);

            _mapWriter = new outputs::XYZ_Map_Writer("out");
        }

        Local_Map::~Local_Map()
        {
            for (const auto& [pointId, mapPoint] : _localPointMap) 
            {
                _mapWriter->add_point(mapPoint._coordinates);
            }

            delete _mapWriter;
        }

        const features::keypoints::KeypointsWithIdStruct Local_Map::get_tracked_keypoints_features(const utils::Pose& lastPose) const
        {
            const size_t numberOfNewKeypoints = _localPointMap.size() + _stagedPoints.size();

            const worldToCameraMatrix& worldToCamera = utils::compute_world_to_camera_transform(lastPose.get_orientation_quaternion(), lastPose.get_position());

            // initialize output structure
            features::keypoints::KeypointsWithIdStruct keypointsWithIds; 

            // TODO: check the efficiency gain of those reserve calls
            keypointsWithIds._ids.reserve(numberOfNewKeypoints);
            keypointsWithIds._keypoints.reserve(numberOfNewKeypoints);

            const static uint refreshFrequency = Parameters::get_keypoint_refresh_frequency();

            // add map points with valid retroprojected coordinates
            for (const auto& [pointId, point]  : _localPointMap)
            {
                assert(pointId == point._id);
                add_point_to_tracked_features(worldToCamera, point, keypointsWithIds, refreshFrequency * 2);
            }
            // add staged points with valid retroprojected coordinates
            for (const auto& [pointId, point] : _stagedPoints)
            {
                assert(pointId == point._id);
                add_point_to_tracked_features(worldToCamera, point, keypointsWithIds, refreshFrequency);
            }
            return keypointsWithIds;
        }

        matches_containers::matchContainer Local_Map::find_feature_matches(const utils::Pose& currentPose, const features::keypoints::Keypoint_Handler& detectedKeypointsObject, const features::primitives::plane_container& detectedPlanes)
        {
            matches_containers::matchContainer matchSets;

            matchSets._points = find_keypoint_matches(currentPose, detectedKeypointsObject);
            matchSets._planes = find_plane_matches(currentPose, detectedPlanes);

            return matchSets;
        }

        void Local_Map::update(const utils::Pose& optimizedPose, const features::keypoints::Keypoint_Handler& keypointObject, const features::primitives::plane_container& detectedPlanes, const matches_containers::match_point_container& outlierMatchedPoints, const matches_containers::match_plane_container& outlierMatchedPlanes)
        {
            // TODO find a better way to display trajectory than just a new map point
            // _mapWriter->add_point(optimizedPose.get_position());

            // Unmatch detected outliers
            mark_outliers_as_unmatched(outlierMatchedPoints);
            mark_outliers_as_unmatched(outlierMatchedPlanes);

            const matrix33& poseCovariance = utils::compute_pose_covariance(optimizedPose);
            const cameraToWorldMatrix& cameraToWorld = utils::compute_camera_to_world_transform(optimizedPose.get_orientation_quaternion(), optimizedPose.get_position());

            // add local map points
            update_local_keypoint_map(cameraToWorld, keypointObject);

            // add staged points to local map
            update_staged_keypoints_map(cameraToWorld, keypointObject);

            // Add unmatched poins to the staged map, to unsure tracking of new features
            add_umatched_keypoints_to_staged_map(poseCovariance, cameraToWorld, keypointObject);

            // add planes to local map
            update_local_plane_map(cameraToWorld, detectedPlanes);

            // add local map points to global map
            update_local_to_global();
        }

        void Local_Map::update_local_to_global() 
        {
            // TODO when we have a global map

        }

        void Local_Map::update_no_pose()
        {
            // add local map points
            update_local_keypoint_map_with_tracking_lost();

            // add staged points to local map
            update_staged_keypoints_map_with_tracking_lost();

            // add planes to local map
            update_local_plane_map_with_tracking_lost();
        }

        void Local_Map::reset()
        {
            _localPointMap.clear();
            _stagedPoints.clear();
        }



        /**
         * PROTECTED
         */

        bool Local_Map::find_match(IMap_Point_With_Tracking& point, const features::keypoints::Keypoint_Handler& detectedKeypointsObject, const worldToCameraMatrix& worldToCamera, matches_containers::match_point_container& matchedPoints, const bool shouldAddMatchToContainer)
        {
            // try to find a match with opticalflow
            int matchIndex = detectedKeypointsObject.get_tracking_match_index(point._id, _isPointMatched);
            if (matchIndex == features::keypoints::INVALID_MATCH_INDEX)
            {
                // No match: try to find match in a window around the point
                utils::ScreenCoordinate2D projectedMapPoint;
                const bool isScreenCoordinatesValid = (point._coordinates).to_screen_coordinates(worldToCamera, projectedMapPoint);
                if (isScreenCoordinatesValid)
                    matchIndex = detectedKeypointsObject.get_match_index(projectedMapPoint, point._descriptor, _isPointMatched);
            }

            if (matchIndex == features::keypoints::INVALID_MATCH_INDEX) {
                //unmatched point
                point.mark_unmatched();
                return false;
            }

            assert(matchIndex >= 0);

            const utils::ScreenCoordinate& matchedScreenpoint = detectedKeypointsObject.get_keypoint(matchIndex);
            if (_isPointMatched[matchIndex])
            {
                //point was already matched
                outputs::log_error("The requested point unique index is already matched");
            }

            if (utils::is_depth_valid(matchedScreenpoint.z()) ) {
                // points with depth measurement
                _isPointMatched[matchIndex] = true;
                // update match index
                point._matchIndex = matchIndex;

                if(shouldAddMatchToContainer)
                {
                    const rgbd_slam::screenCoordinateCovariance& screenCovariance = utils::get_screen_point_covariance(point._coordinates, point.get_covariance_matrix());
                    //consider only the diagonal part of the matrix: it is the 2D variance en x/y in screen space
                    const vector2& screenPointCovariance(screenCovariance.diagonal().head(2));
                    matchedPoints.emplace(matchedPoints.end(), matchedScreenpoint, point._coordinates, screenPointCovariance, point._id);
                }
                return true;
            }
            else {
                // 2D point
                _isPointMatched[matchIndex] = true;
                // update index and screen coordinates 
                point._matchIndex = matchIndex;

                if(shouldAddMatchToContainer)
                {
                    const rgbd_slam::screenCoordinateCovariance& screenCovariance = utils::get_screen_point_covariance(point._coordinates, point.get_covariance_matrix());
                    //consider only the diagonal part of the matrix: it is the 2D variance en x/y in screen space
                    const vector2& screenPointCovariance(screenCovariance.diagonal().head(2));
                    matchedPoints.emplace(matchedPoints.end(), matchedScreenpoint, point._coordinates, screenPointCovariance, point._id);
                }
                return true;
            }
            return false;
        }

        bool Local_Map::find_match(MapPlane& mapPlane, const features::primitives::plane_container& detectedPlanes, const planeWorldToCameraMatrix& worldToCamera, matches_containers::match_plane_container& matchedPlanes)
        {
            // project plane in camera space
            const utils::PlaneCameraCoordinates& projectedPlane = mapPlane._parametrization.to_camera_coordinates(worldToCamera);
            for(const auto& [planeId, shapePlane] : detectedPlanes)
            {
                assert(planeId == shapePlane.get_id());
                assert(planeId != UNMATCHED_PRIMITIVE_ID);

                if (not _unmatchedPlaneIds.contains(planeId))
                    // Does not allow multiple removal of a single match
                    // TODO: change this
                    continue;

                if(shapePlane.is_similar(mapPlane._shapeMask, projectedPlane)) 
                {
                    mapPlane._matchedPlane.mark_matched(planeId);
                    // TODO: replace nullptr by the plane covariance in camera space
                    matchedPlanes.emplace(matchedPlanes.end(), shapePlane._parametrization, mapPlane._parametrization, nullptr, mapPlane._id);

                    _unmatchedPlaneIds.erase(planeId);
                    return true;
                }
            }

            return false;
        }

        void Local_Map::update_local_plane_map(const cameraToWorldMatrix& cameraToWorld, const features::primitives::plane_container& detectedPlanes)
        {
            std::set<size_t> planesToRemove;
            const planeCameraToWorldMatrix& planeCameraToWorld = utils::compute_plane_camera_to_world_matrix(cameraToWorld);

            // Update planes
            for (auto& [planeId, mapPlane] : _localPlaneMap)
            {
                if (mapPlane._matchedPlane.is_matched())
                {
                    const size_t matchedPlaneId = mapPlane._matchedPlane._matchId;
                    assert(matchedPlaneId != UNMATCHED_PRIMITIVE_ID);
                    assert(detectedPlanes.contains(matchedPlaneId));

                    const features::primitives::Plane& detectedPlane = detectedPlanes.at(matchedPlaneId);
                    // TODO update plane
                    mapPlane._parametrization = detectedPlane._parametrization.to_world_coordinates(planeCameraToWorld);
                    mapPlane._shapeMask = detectedPlane.get_shape_mask();
                }
                else if (mapPlane._matchedPlane.is_lost())
                {
                    // add to planes to remove
                    planesToRemove.emplace(planeId);
                }
            }

            // Remove umatched
            for(const size_t planeId : planesToRemove)
            {
                _localPlaneMap.erase(planeId);
            }

            // add unmatched planes to local map
            for(const features::primitives::planeId& unmatchedDetectedPlaneId : _unmatchedPlaneIds)
            {
                assert(detectedPlanes.contains(unmatchedDetectedPlaneId));

                const features::primitives::Plane& detectedPlane = detectedPlanes.at(unmatchedDetectedPlaneId);
                
                MapPlane newMapPlane;
                newMapPlane._parametrization = detectedPlane._parametrization.to_world_coordinates(planeCameraToWorld);
                newMapPlane._shapeMask = detectedPlane.get_shape_mask();

                _localPlaneMap.emplace(newMapPlane._id, newMapPlane);
            }

            _unmatchedPlaneIds.clear();
        }

        void Local_Map::update_local_plane_map_with_tracking_lost()
        {
            std::set<size_t> planesToRemove;
            // Update planes
            for (auto& [planeId, mapPlane] : _localPlaneMap)
            {
                if (mapPlane._matchedPlane.is_lost())
                {
                    // add to planes to remove
                    planesToRemove.emplace(planeId);
                }
            }

            // Remove umatched
            for(const size_t planeId : planesToRemove)
            {
                _localPlaneMap.erase(planeId);
            }

            _unmatchedPlaneIds.clear();
        }

        void Local_Map::update_point_match_status(IMap_Point_With_Tracking& mapPoint, const features::keypoints::Keypoint_Handler& keypointObject, const cameraToWorldMatrix& cameraToWorld)
        {
            if (mapPoint.is_matched())
            {
                assert(mapPoint._matchIndex >= 0);
                const size_t matchedPointIndex = mapPoint._matchIndex;
                assert(matchedPointIndex < keypointObject.get_keypoint_count());

                // get match coordinates, transform them to world coordinates
                const utils::ScreenCoordinate& matchedPointCoordinates = keypointObject.get_keypoint(matchedPointIndex);

                if(utils::is_depth_valid(matchedPointCoordinates.z()))
                {
                    // transform screen point to world point
                    const utils::WorldCoordinate& worldPointCoordinates = matchedPointCoordinates.to_world_coordinates(cameraToWorld);
                    // get a measure of the estimated variance of the new world point
                    const cameraCoordinateCovariance& cameraPointCovariance = utils::get_camera_point_covariance(matchedPointCoordinates);

                    // update this map point errors & position
                    mapPoint.update_matched(worldPointCoordinates, cameraPointCovariance.base());

                    // If a new descriptor is available, update it
                    if (keypointObject.is_descriptor_computed(matchedPointIndex))
                        mapPoint._descriptor = keypointObject.get_descriptor(matchedPointIndex);

                    // End of the function
                    return;
                }
                else
                {
#if 0
                    // inefficient...
                    const worldToCameraMatrix& worldToCamera = utils::compute_world_to_camera_transform(previousCameraToWorldMatrix);
                    vector2 previousPointScreenCoordinates;
                    const bool isTransformationValid = (mapPoint._coordinates).to_screen_coordinates(worldToCamera, previousPointScreenCoordinates);
                    if (isTransformationValid)
                    {
                        worldPointCoordinates triangulatedPoint;
                        const bool isTriangulationValid = tracking::Triangulation::triangulate(previousCameraToWorld, cameraToWorld, previousPointScreenCoordinates, matchedPointCoordinates, triangulatedPoint);
                        // update the match
                        if (isTriangulationValid)
                        {
                            //std::cout << "udpate with triangulation " << triangulatedPoint.transpose() << " " << mapPoint._coordinates.transpose() << std::endl;
                            // get a measure of the estimated variance of the new world point
                            //const matrix33& cameraPointCovariance = utils::get_triangulated_point_covariance(triangulatedPoint, get_screen_point_covariance(triangulatedPoint.z())));
                            const cameraCoordinateCovariance& cameraPointCovariance = utils::get_camera_point_covariance(vector2(triangulatedPoint.x(), triangulatedPoint.y()), triangulatedPoint.z(), get_screen_point_covariance(triangulatedPoint.z()));

                            // update this map point errors & position
                            mapPoint.update_matched(triangulatedPoint, cameraPointCovariance);

                            // If a new descriptor is available, update it
                            if (keypointObject.is_descriptor_computed(matchedPointIndex))
                                mapPoint._descriptor = keypointObject.get_descriptor(matchedPointIndex);
                            return;
                        }
                    }
#endif
                    return;
                }
            }

            // point is unmatched
            mapPoint.update_unmatched();
        }

        void Local_Map::update_local_keypoint_map(const cameraToWorldMatrix& cameraToWorld, const features::keypoints::Keypoint_Handler& keypointObject)
        {
            // use this precprocessor directiv if you observe a lot of duplicated points in the local map
            #ifdef REMOVE_DUPLICATE_STAGED_POINTS
            const static double maximumMatchDistance = Parameters::get_maximum_match_distance();
            const static float searchDiameter = Parameters::get_search_matches_distance();
            cv::Mat stagedPointDescriptors;
            std::unordered_map<size_t, size_t> indexToId;
            size_t index = 0;
            for(const auto& [pointId, mapPoint] : _stagedPoints)
            {
                indexToId[index++] = pointId;
                if (stagedPointDescriptors.rows == 0)
                    stagedPointDescriptors = mapPoint._descriptor;
                else
                    cv::vconcat(mapPoint._descriptor, stagedPointDescriptors, stagedPointDescriptors);
            }
            cv::Ptr<cv::DescriptorMatcher> featuresMatcher = cv::Ptr<cv::BFMatcher>(new cv::BFMatcher());
            #endif

            point_map_container::iterator pointMapIterator = _localPointMap.begin();
            while(pointMapIterator != _localPointMap.end())
            {
                // Update the matched/unmatched status
                Map_Point& mapPoint = pointMapIterator->second;
                assert(pointMapIterator->first == mapPoint._id);

                // update the point match status (matched/unmatched)
                update_point_match_status(mapPoint, keypointObject, cameraToWorld);

                if (mapPoint.is_lost()) {
                    // write to file
                    _mapWriter->add_point(mapPoint._coordinates);

                    // Remove useless point
                    pointMapIterator = _localPointMap.erase(pointMapIterator);
                }
                else
                {
                    #ifdef REMOVE_DUPLICATE_STAGED_POINTS
                    // try to find a point in the staged map, that is a duplicate of the this map point
                    std::vector<std::vector<cv::DMatch>> knnMatches;
                    featuresMatcher->knnMatch(mapPoint._descriptor, stagedPointDescriptors, knnMatches, 2);

                    const std::vector<cv::DMatch>& match = knnMatches[0];
                    if (match.size() >= 1) {
                        if (match.size() == 1 or match[0].distance < maximumMatchDistance * match[1].distance)
                        {
                            const size_t stagedId = indexToId[match[0].trainIdx];
                            Staged_Point& matchedStagedPoint = _stagedPoints.find(stagedId)->second;
                            if (matchedStagedPoint._coordinates.get_distance(mapPoint._coordinates) < searchDiameter)
                            {
                                //if (not mapPoint.is_matched()) mapPoint._matchIndex = matchedStagedPoint._matchIndex;
                                //else mapPoint.update_matched(matchedStagedPoint._coordinates, matchedStagedPoint.get_covariance_matrix());
                                matchedStagedPoint.mark_unmatched();
                            }
                        }
                    }
                    #endif
                    ++pointMapIterator;
                }
            }
        }

        void Local_Map::update_local_keypoint_map_with_tracking_lost()
        {
            point_map_container::iterator pointMapIterator = _localPointMap.begin();
            while(pointMapIterator != _localPointMap.end())
            {
                // Update the matched/unmatched status
                Map_Point& mapPoint = pointMapIterator->second;
                assert(pointMapIterator->first == mapPoint._id);

                mapPoint.update_unmatched();

                if (mapPoint.is_lost()) {
                    // write to file
                    _mapWriter->add_point(mapPoint._coordinates);

                    // Remove useless point
                    pointMapIterator = _localPointMap.erase(pointMapIterator);
                }
                else
                {
                    ++pointMapIterator;
                }
            }
        }

        void Local_Map::update_staged_keypoints_map(const cameraToWorldMatrix& cameraToWorld, const features::keypoints::Keypoint_Handler& keypointObject)
        {
            // Add correct staged points to local map
            staged_point_container::iterator stagedPointIterator = _stagedPoints.begin();
            while(stagedPointIterator != _stagedPoints.end())
            {
                Staged_Point& stagedPoint = stagedPointIterator->second;
                assert(stagedPointIterator->first == stagedPoint._id);

                // Update the matched/unmatched status
                update_point_match_status(stagedPoint, keypointObject, cameraToWorld);

                if (stagedPoint.should_add_to_local_map())
                {
                    const utils::WorldCoordinate& stagedPointCoordinates = stagedPoint._coordinates;
                    assert(not std::isnan(stagedPointCoordinates.x()) and not std::isnan(stagedPointCoordinates.y()) and not std::isnan(stagedPointCoordinates.z()));
                    // Add to local map, remove from staged points, with a copy of the id affected to the local map
                    _localPointMap.emplace(
                            stagedPoint._id,
                            Map_Point(stagedPointCoordinates, stagedPoint.get_covariance_matrix(), stagedPoint._descriptor, stagedPoint._id)
                            );
                    _localPointMap.at(stagedPoint._id)._matchIndex = stagedPoint._matchIndex;
                    stagedPointIterator = _stagedPoints.erase(stagedPointIterator);
                }
                else if (stagedPoint.should_remove_from_staged())
                {
                    // Remove from staged points
                    stagedPointIterator = _stagedPoints.erase(stagedPointIterator);
                }
                else
                {
                    // Increment
                    ++stagedPointIterator;
                }
            }
        }

        void Local_Map::update_staged_keypoints_map_with_tracking_lost()
        {
            // Add correct staged points to local map
            staged_point_container::iterator stagedPointIterator = _stagedPoints.begin();
            while(stagedPointIterator != _stagedPoints.end())
            {
                Staged_Point& stagedPoint = stagedPointIterator->second;
                assert(stagedPointIterator->first == stagedPoint._id);

                // Update the matched/unmatched status
                stagedPoint.update_unmatched();

                if (stagedPoint.should_remove_from_staged())
                {
                    // Remove from staged points
                    stagedPointIterator = _stagedPoints.erase(stagedPointIterator);
                }
                else
                {
                    // Increment
                    ++stagedPointIterator;
                }
            }
        }

        void Local_Map::add_umatched_keypoints_to_staged_map(const matrix33& poseCovariance, const cameraToWorldMatrix& cameraToWorld, const features::keypoints::Keypoint_Handler& keypointObject)
        {
            // Add all unmatched points to staged point container 
            const size_t keypointVectorSize = _isPointMatched.size();
            for(unsigned int i = 0; i < keypointVectorSize; ++i)
            {
                if (not _isPointMatched[i]) {
                    // TODO remove this condition when we will compute the descriptor for optical flow points 
                    if(! keypointObject.is_descriptor_computed(i))
                    {
                        continue;
                    }

                    // TODO: add points with invalid depth to stagged map  ?
                    const utils::ScreenCoordinate& screenPoint = keypointObject.get_keypoint(i);
                    if (not utils::is_depth_valid(screenPoint.z()))
                    {
                        continue;
                    }

                    const utils::WorldCoordinate& worldPoint = screenPoint.to_world_coordinates(cameraToWorld);
                    assert(not std::isnan(worldPoint.x()) and not std::isnan(worldPoint.y()) and not std::isnan(worldPoint.z()));

                    const cameraCoordinateCovariance& cameraPointCovariance = utils::get_camera_point_covariance(screenPoint);

                    Staged_Point newStagedPoint(worldPoint, cameraPointCovariance + poseCovariance, keypointObject.get_descriptor(i));
                    // add to staged map
                    _stagedPoints.emplace(
                            newStagedPoint._id,
                            newStagedPoint);
                }
            }

        }

        void Local_Map::draw_point_on_image(const IMap_Point_With_Tracking& mapPoint, const worldToCameraMatrix& worldToCameraMatrix, const cv::Scalar& pointColor, cv::Mat& debugImage, const size_t radius)
        {
            if (mapPoint.is_matched())
            {
                utils::ScreenCoordinate2D screenPoint;
                const bool isCoordinatesValid = (mapPoint._coordinates).to_screen_coordinates(worldToCameraMatrix, screenPoint);

                //Map Point are green 
                if (isCoordinatesValid)
                {
                    cv::circle(debugImage, cv::Point(static_cast<int>(screenPoint.x()), static_cast<int>(screenPoint.y())), radius, pointColor, -1);
                }
            }
        }

        void Local_Map::draw_planes_on_image(const worldToCameraMatrix& worldToCamera, cv::Mat& debugImage) const
        {
            assert(not debugImage.empty());

            const cv::Size& debugImageSize = debugImage.size();
            const uint imageWidth = debugImageSize.width;

            const double maskAlpha = 0.3;
            // 20 pixels
            const uint bandSize = 20;
            const uint placeInBand = bandSize * 0.75;

            std::stringstream textPoints;
            textPoints << "Points:" << _localPointMap.size();
            const int plointLabelPosition = static_cast<int>(imageWidth * 0.15);
            cv::putText(debugImage, textPoints.str(), cv::Point(plointLabelPosition, placeInBand), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 1));

            std::stringstream text1;
            const double planeOffset = 0.35;
            text1 << "Planes:";
            const int planeLabelPosition = static_cast<int>(imageWidth * planeOffset);
            cv::putText(debugImage, text1.str(), cv::Point(planeLabelPosition, placeInBand), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 1));

            std::stringstream text2;
            const double cylinderOffset = 0.70;
            text2 << "Cylinders:";
            const int cylinderLabelPosition = static_cast<int>(imageWidth * 0.70);
            cv::putText(debugImage, text2.str(), cv::Point(cylinderLabelPosition, placeInBand), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 1));

            // Tracking variables
            uint cylinderCount = 0;
            uint planeCount = 0;
            std::set<size_t> alreadyDisplayedIds;

            if (_localPlaneMap.size() == 0)
                return;

            cv::Mat allPlaneMasks = cv::Mat::zeros(debugImageSize, debugImage.type());
            for(const auto& [planeId, mapPlane]: _localPlaneMap)
            {
                if (not mapPlane._matchedPlane.is_matched())
                    continue;

                const cv::Scalar& planeColor = mapPlane._color;

                cv::Mat planeMask;
                // Resize with no interpolation
                cv::resize(mapPlane._shapeMask * 255, planeMask, debugImageSize, 0, 0, cv::INTER_NEAREST);
                cv::cvtColor(planeMask, planeMask, cv::COLOR_GRAY2BGR);
                assert(planeMask.size == debugImage.size);
                assert(planeMask.type() == debugImage.type());

                // merge with debug image
                planeMask.setTo(planeColor, planeMask);
                allPlaneMasks += planeMask;

                // Add color codes in label bar
                if (alreadyDisplayedIds.contains(planeId))
                    continue;   // already shown

                alreadyDisplayedIds.insert(planeId);

                double labelPosition = imageWidth;
                uint finalPlaceInBand = placeInBand;

                // plane
                labelPosition *= planeOffset;
                finalPlaceInBand *= planeCount;
                ++planeCount;

                    /*case features::primitives::PrimitiveType::Cylinder:
                        labelPosition *= cylinderOffset;
                        finalPlaceInBand *= cylinderCount;
                        ++cylinderCount;
                        break;*/

                if (labelPosition >= 0)
                {
                    // make a
                    const uint labelSquareSize = bandSize * 0.5;
                    cv::rectangle(debugImage, 
                            cv::Point(static_cast<int>(labelPosition + 80 + finalPlaceInBand), 6),
                            cv::Point(static_cast<int>(labelPosition + 80 + labelSquareSize + finalPlaceInBand), 6 + labelSquareSize), 
                            planeColor,
                            -1);
                }
            }

            cv::addWeighted(debugImage, (1 - maskAlpha), allPlaneMasks, maskAlpha, 0.0, debugImage);
        }

        void Local_Map::get_debug_image(const utils::Pose& camPose, const bool shouldDisplayStaged, const bool shouldDisplayPlaneMasks, cv::Mat& debugImage)  const
        {
            const worldToCameraMatrix& worldToCamMatrix = utils::compute_world_to_camera_transform(camPose.get_orientation_quaternion(), camPose.get_position());

            // Display planes
            if (shouldDisplayPlaneMasks)
                draw_planes_on_image(worldToCamMatrix, debugImage);

            // Display stagged points if needed
            if (shouldDisplayStaged)
            {
                for (const auto& [pointId, stagedPoint] : _stagedPoints) {
                    assert(pointId == stagedPoint._id);

                    const cv::Scalar pointColor = (stagedPoint.is_matched()) ? cv::Scalar(0, 200, 255) : cv::Scalar(0, 255, 0);
                    draw_point_on_image(stagedPoint, worldToCamMatrix, pointColor, debugImage, 2);
                }
            }
            // Display true map points
            for (const auto& [pointId, mapPoint] : _localPointMap) {
                assert(pointId == mapPoint._id);
                draw_point_on_image(mapPoint, worldToCamMatrix, mapPoint._color, debugImage);
            }
        }

        matches_containers::match_point_container Local_Map::find_keypoint_matches(const utils::Pose& currentPose, const features::keypoints::Keypoint_Handler& detectedKeypointsObject)
        {
            // will be used to detect new keypoints for the stagged map
            _isPointMatched.assign(detectedKeypointsObject.get_keypoint_count(), false);
            matches_containers::match_point_container matchedPoints; 

            const worldToCameraMatrix& worldToCamera = utils::compute_world_to_camera_transform(currentPose.get_orientation_quaternion(), currentPose.get_position());

            // Try to find matches in local map
            for (auto& [pointId, mapPoint] : _localPointMap) 
            {
                assert(pointId == mapPoint._id);
                find_match(mapPoint, detectedKeypointsObject, worldToCamera, matchedPoints);
            }

            // if we have enough points from local map to run the optimization, no need to add the staged points
            // Still, we need to try and match them to insure tracking and new map points
            const static uint minimumPointsForOptimization = Parameters::get_minimum_point_count_for_optimization() * 3;   // TODO: Why 3 ? seems about right to be sure to have enough points for the optimization process... 
            const bool shouldUseStagedPoints = matchedPoints.size() < minimumPointsForOptimization;

            // Try to find matches in staged points
            for(auto& [pointId, stagedPoint] : _stagedPoints)
            {
                assert(pointId == stagedPoint._id);
                find_match(stagedPoint, detectedKeypointsObject, worldToCamera, matchedPoints, shouldUseStagedPoints);
            }

            return matchedPoints;
        }

        matches_containers::match_plane_container Local_Map::find_plane_matches(const utils::Pose& currentPose, const features::primitives::plane_container& detectedPlanes)
        {
            _unmatchedPlaneIds.clear();
            // Fill in all features ids
            for(const auto& [planeId, shapePlane] : detectedPlanes)
            {
                assert(planeId == shapePlane.get_id());
                assert(planeId != UNMATCHED_PRIMITIVE_ID);
                if (not _unmatchedPlaneIds.insert(planeId).second)
                {
                    // This element was already in the map
                    outputs::log_error("A plane index was already maintained in set");
                }
            }

            // Compute a world to camera transformation matrix
            const worldToCameraMatrix& worldToCamera = utils::compute_world_to_camera_transform(currentPose.get_orientation_quaternion(), currentPose.get_position());
            const planeWorldToCameraMatrix& planeWorldToCamera = utils::compute_plane_world_to_camera_matrix(worldToCamera);

            // Search for matches
            matches_containers::match_plane_container matchedPlaneContainer;
            for(auto& [planeId, mapPlane] : _localPlaneMap)
            {
                if (not find_match(mapPlane, detectedPlanes, planeWorldToCamera, matchedPlaneContainer))
                {
                    // Mark as unmatched
                    mapPlane._matchedPlane.mark_unmatched();
                }
            }

            return matchedPlaneContainer;
        }


        void Local_Map::mark_outliers_as_unmatched(const matches_containers::match_point_container& outlierMatchedPoints)
        {
            // Mark outliers as unmatched
            for (const matches_containers::PointMatch& match : outlierMatchedPoints)
            {
                const bool isOutlierRemoved = mark_point_with_id_as_unmatched(match._idInMap);
                // If no points were found, this is bad. A match marked as outliers must be in the local map or staged points
                if(not isOutlierRemoved)
                {
                    outputs::log_error("Could not find the target point with id " + std::to_string(match._idInMap));
                }
            }
        }

        void Local_Map::mark_outliers_as_unmatched(const matches_containers::match_plane_container& outlierMatchedPlanes)
        {
            // Mark outliers as unmatched
            for (const matches_containers::PlaneMatch& match : outlierMatchedPlanes)
            {
                // Check if id is in local map
                const size_t planeId = match._idInMap;
                plane_map_container::iterator planeMapIterator = _localPlaneMap.find(planeId);
                if (planeMapIterator != _localPlaneMap.end())
                {
                    MapPlane& mapPlane = planeMapIterator->second;
                    assert(mapPlane._id == planeId);
                    // add detected plane id to unmatched set
                    _unmatchedPlaneIds.insert(mapPlane._matchedPlane._matchId);
                    // unmatch
                    mapPlane._matchedPlane.mark_unmatched();
                }
                else
                {
                    outputs::log_error("Could not find the target plane with id " + std::to_string(planeId));
                }
            }
        }

        bool Local_Map::mark_point_with_id_as_unmatched(const size_t pointId)
        {
            // Check if id is in local map
            point_map_container::iterator pointMapIterator = _localPointMap.find(pointId);
            if (pointMapIterator != _localPointMap.end())
            {
                Map_Point& mapPoint = pointMapIterator->second;
                assert(mapPoint._id == pointId);
                mark_point_with_id_as_unmatched(mapPoint);
                return true;
            }

            // Check if id is in staged points
            staged_point_container::iterator stagedPointIterator = _stagedPoints.find(pointId);
            if (stagedPointIterator != _stagedPoints.end())
            {
                Staged_Point& stagedPoint = stagedPointIterator->second;
                assert(stagedPoint._id == pointId);
                mark_point_with_id_as_unmatched(stagedPoint);
                return true;
            }

            // point associated with id was not find
            return false;
        }

        void Local_Map::mark_point_with_id_as_unmatched(IMap_Point_With_Tracking& point)
        {
            const int matchIndex = point._matchIndex;
            assert(matchIndex >= 0 and matchIndex < static_cast<int>(_isPointMatched.size()));

            // Mark point as unmatched
            _isPointMatched[matchIndex] = false;
            point.mark_unmatched();
        }

    }   /* map_management */
}   /* rgbd_slam */
