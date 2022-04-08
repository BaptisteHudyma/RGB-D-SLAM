#include "local_map.hpp"

#include "parameters.hpp"
#include "triangulation.hpp"
#include "camera_transformation.hpp"
#include "covariances.hpp"
#include "logger.hpp"

namespace rgbd_slam {
    namespace map_management {

        /**
         * LOCAL UTILS FUNCTIONS
         */

        /**
         * \brief My add a point to the tracked feature object, used to add optical flow tracking
         */
        void add_point_to_tracked_features(const IMap_Point_With_Tracking& mapPoint, features::keypoints::KeypointsWithIdStruct& keypointsWithIds)
        {
            const vector3& coordinates = mapPoint._coordinates;
            assert(not std::isnan(coordinates.x()) and not std::isnan(coordinates.y()) and not std::isnan(coordinates.z()));
            if (mapPoint._matchedScreenPoint.is_matched())
            {
                // use previously known screen coordinates
                keypointsWithIds._keypoints.push_back(cv::Point2f(mapPoint._matchedScreenPoint._screenCoordinates.x(), mapPoint._matchedScreenPoint._screenCoordinates.y()));
                keypointsWithIds._ids.push_back(mapPoint._id);
            }
        }

        /**
         * LOCAL MAP MEMBERS
         */

        Local_Map::Local_Map()
        {
            // Check constants
            assert(features::keypoints::INVALID_MAP_POINT_ID == INVALID_POINT_UNIQ_ID);

            set_color_vector();

            _mapWriter = new utils::XYZ_Map_Writer("out");
        }

        Local_Map::~Local_Map()
        {
            for (const auto& [pointId, mapPoint] : _localPointMap) 
            {
                _mapWriter->add_point(mapPoint._coordinates);
            }

            delete _mapWriter;
        }

        bool Local_Map::find_match(IMap_Point_With_Tracking& point, const features::keypoints::Keypoint_Handler& detectedKeypointsObject, const matrix44& worldToCamMatrix, matches_containers::match_point_container& matchedPoints)
        {
            int matchIndex = detectedKeypointsObject.get_tracking_match_index(point._id, _isPointMatched);
            if (matchIndex == features::keypoints::INVALID_MATCH_INDEX)
            {
                vector2 projectedMapPoint;
                const bool isScreenCoordinatesValid = utils::world_to_screen_coordinates(point._coordinates, worldToCamMatrix, projectedMapPoint);
                if (isScreenCoordinatesValid)
                    matchIndex = detectedKeypointsObject.get_match_index(projectedMapPoint, point._descriptor, _isPointMatched);
            }

            if (matchIndex == features::keypoints::INVALID_MATCH_INDEX) {
                //unmatched point
                point._matchedScreenPoint.mark_unmatched();
                return false;
            }

            const double screenPointDepth = detectedKeypointsObject.get_depth(matchIndex);
            if (utils::is_depth_valid(screenPointDepth) ) {
                // points with depth measurement
                _isPointMatched[matchIndex] = true;

                // update index and screen coordinates 
                MatchedScreenPoint match;
                match._screenCoordinates << detectedKeypointsObject.get_keypoint(matchIndex), screenPointDepth;
                match._matchIndex = matchIndex;
                point._matchedScreenPoint = match;

                matchedPoints.emplace(matchedPoints.end(), match._screenCoordinates, point._coordinates, point._id);
                return true;
            }
            else {
                // 2D point
                _isPointMatched[matchIndex] = true;

                // update index and screen coordinates 
                MatchedScreenPoint match;
                match._screenCoordinates << detectedKeypointsObject.get_keypoint(matchIndex), 0;
                match._matchIndex = matchIndex;
                point._matchedScreenPoint = match;

                matchedPoints.emplace(matchedPoints.end(), match._screenCoordinates, point._coordinates, point._id);
                return true;
            }
            return false;
        }

        bool Local_Map::find_match(Primitive& mapPrimitive, const features::primitives::primitive_container& detectedPrimitives, const matrix44& worldToCameraMatrix, matches_containers::match_primitive_container& matchedPrimitives)
        {
            // TODO: convert mapPrimitive to camera space
            for(const auto& [primitiveId, shapePrimitive] : detectedPrimitives)
            {
                assert(primitiveId == shapePrimitive->get_id());
                assert(primitiveId != UNMATCHED_PRIMITIVE_ID);

                if (not _unmatchedPrimitiveIds.contains(primitiveId))
                    // Does not allow multiple removal of a single match
                    // TODO: change this
                    continue;

                if(mapPrimitive._primitive->is_similar(shapePrimitive)) 
                {
                    mapPrimitive._matchedPrimitive._matchId = primitiveId;
                    matchedPrimitives.emplace(matchedPrimitives.end(), shapePrimitive->_normal, mapPrimitive._primitive->_normal);

                    _unmatchedPrimitiveIds.erase(primitiveId);
                    return true;
                }
            }

            return false;
        }

        matches_containers::match_point_container Local_Map::find_keypoint_matches(const utils::Pose& currentPose, const features::keypoints::Keypoint_Handler& detectedKeypointsObject)
        {
            // will be used to detect new keypoints for the stagged map
            _isPointMatched.assign(detectedKeypointsObject.get_keypoint_count(), false);
            matches_containers::match_point_container matchedPoints; 

            const matrix44& worldToCamMatrix = utils::compute_world_to_camera_transform(currentPose.get_orientation_quaternion(), currentPose.get_position());

            // Try to find matches in local map
            for (auto& [pointId, mapPoint] : _localPointMap) 
            {
                assert(pointId == mapPoint._id);
                find_match(mapPoint, detectedKeypointsObject, worldToCamMatrix, matchedPoints);
            }

            // Try to find matches in staged points
            for(auto& [pointId, stagedPoint] : _stagedPoints)
            {
                assert(pointId == stagedPoint._id);
                find_match(stagedPoint, detectedKeypointsObject, worldToCamMatrix, matchedPoints);
            }

            return matchedPoints;
        }

        matches_containers::match_primitive_container Local_Map::find_primitive_matches(const utils::Pose& currentPose, const features::primitives::primitive_container& detectedPrimitives)
        {
            _unmatchedPrimitiveIds.clear();
            // Fill in all features ids
            for(const auto& [primitiveId, shapePrimitive] : detectedPrimitives)
            {
                assert(primitiveId == shapePrimitive->get_id());
                assert(primitiveId != UNMATCHED_PRIMITIVE_ID);
                if (not _unmatchedPrimitiveIds.insert(primitiveId).second)
                {
                    // This element was already in the map
                    utils::log_error("A primitive index was already maintained in set");
                }
            }

            // Compute a world to camera transformation matrix
            const matrix44& worldToCameraMatrix = utils::compute_world_to_camera_transform(currentPose.get_orientation_quaternion(), currentPose.get_position());

            // Search for matches
            matches_containers::match_primitive_container matchedPrimitiveContainer;
            for(auto& [primitiveId, mapPrimitive] : _localPrimitiveMap)
            {
                if (not find_match(mapPrimitive, detectedPrimitives, worldToCameraMatrix, matchedPrimitiveContainer))
                {
                    // Mark as unmatched
                    mapPrimitive._matchedPrimitive.mark_unmatched();
                }
            }

            return matchedPrimitiveContainer;
        }

        void Local_Map::update(const utils::Pose& previousPose, const utils::Pose& optimizedPose, const features::keypoints::Keypoint_Handler& keypointObject, const features::primitives::primitive_container& detectedPrimitives, const matches_containers::match_point_container& outlierMatchedPoints)
        {
            // TODO find a better way to display trajectory than just a new map point
            _mapWriter->add_point(optimizedPose.get_position());

            // Unmatch detected outliers
            mark_outliers_as_unmatched(outlierMatchedPoints);

            const matrix33& poseCovariance = utils::compute_pose_covariance(optimizedPose);
            const matrix44& previousCameraToWorldMatrix = utils::compute_camera_to_world_transform(previousPose.get_orientation_quaternion(), previousPose.get_position());
            const matrix44& cameraToWorldMatrix = utils::compute_camera_to_world_transform(optimizedPose.get_orientation_quaternion(), optimizedPose.get_position());

            // add local map points
            update_local_keypoint_map(poseCovariance, previousCameraToWorldMatrix, cameraToWorldMatrix, keypointObject);

            // add staged points to local map
            update_staged_keypoints_map(poseCovariance, previousCameraToWorldMatrix, cameraToWorldMatrix, keypointObject);

            // Add unmatched poins to the staged map, to unsure tracking of new features
            add_umatched_keypoints_to_staged_map(poseCovariance, cameraToWorldMatrix, keypointObject);

            // add primitives to local map
            update_local_primitive_map(cameraToWorldMatrix, cameraToWorldMatrix, detectedPrimitives);

            // add local map points to global map
            update_local_to_global();
        }

        void Local_Map::update_local_primitive_map(const matrix44& previousCameraToWorldMatrix, const matrix44& cameraToWorldMatrix, const features::primitives::primitive_container& detectedPrimitives)
        {
            std::set<size_t> primitivesToRemove;

            // Update primitives
            for (auto& [primitiveId, mapPrimitive] : _localPrimitiveMap)
            {
                if (mapPrimitive._matchedPrimitive.is_matched())
                {
                    const uchar primitiveId = mapPrimitive._matchedPrimitive._matchId;
                    assert(primitiveId != UNMATCHED_PRIMITIVE_ID);
                    assert(detectedPrimitives.contains(primitiveId));

                    // TODO update primitive 
                    mapPrimitive._primitive->set_shape_mask(detectedPrimitives.at(primitiveId)->get_shape_mask());
                }
                else
                {
                    // add to primitives to remove
                    primitivesToRemove.emplace(primitiveId);
                }   
            }

            // Remove umatched
            for(const size_t primitiveId : primitivesToRemove)
                _localPrimitiveMap.erase(primitiveId);

            // add unmatched primitives to local map
            for(const uchar& unmatchedDetectedPrimitiveId : _unmatchedPrimitiveIds)
            {
                assert(detectedPrimitives.contains(unmatchedDetectedPrimitiveId));

                const features::primitives::primitive_uniq_ptr& detectedPrimitive = detectedPrimitives.at(unmatchedDetectedPrimitiveId);
                Primitive newMapPrimitive(detectedPrimitive);

                _localPrimitiveMap.emplace(newMapPrimitive._id, newMapPrimitive);
            }

            _unmatchedPrimitiveIds.clear();
        }

        void Local_Map::update_point_match_status(IMap_Point_With_Tracking& mapPoint, const matrix33& poseCovariance, const features::keypoints::Keypoint_Handler& keypointObject, const matrix44& previousCameraToWorldMatrix, const matrix44& cameraToWorldMatrix)
        {
            if (mapPoint._matchedScreenPoint.is_matched())
            {
                assert(mapPoint._matchedScreenPoint._matchIndex >= 0);

                const size_t matchedPointIndex = mapPoint._matchedScreenPoint._matchIndex;
                assert(matchedPointIndex < keypointObject.get_keypoint_count()); 

                // get match coordinates, transform them to world coordinates
                const vector2& matchedPointCoordinates = keypointObject.get_keypoint(matchedPointIndex);
                const double matchedPointDepth = keypointObject.get_depth(matchedPointIndex);

                if(utils::is_depth_valid(matchedPointDepth))
                {
                    // transform screen point to world point
                    const vector3& newCoordinates = utils::screen_to_world_coordinates(matchedPointCoordinates.x(), matchedPointCoordinates.y(), matchedPointDepth, cameraToWorldMatrix);
                    // get a measure of the estimated variance of the new world point
                    const matrix33& worldPointCovariance = utils::get_world_point_covariance(matchedPointCoordinates, matchedPointDepth, utils::get_screen_point_covariance(matchedPointCoordinates, matchedPointDepth));

                    // update this map point errors & position
                    mapPoint.update_matched(newCoordinates, worldPointCovariance + poseCovariance);

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
                    const matrix44& worldToCameraMatrix = utils::compute_world_to_camera_transform(previousCameraToWorldMatrix);
                    vector2 previousPointScreenCoordinates;
                    const bool isTransformationValid = utils::world_to_screen_coordinates(mapPoint._coordinates, worldToCameraMatrix, previousPointScreenCoordinates);
                    if (isTransformationValid)
                    {
                        vector3 triangulatedPoint;
                        const bool isTriangulationValid = utils::Triangulation::triangulate(previousCameraToWorldMatrix, cameraToWorldMatrix, previousPointScreenCoordinates, matchedPointCoordinates, triangulatedPoint);
                        // update the match
                        if (isTriangulationValid)
                        {
                            //std::cout << "udpate with triangulation " << triangulatedPoint.transpose() << " " << mapPoint._coordinates.transpose() << std::endl;
                            // get a measure of the estimated variance of the new world point
                            //const matrix33& worldPointCovariance = utils::get_triangulated_point_covariance(triangulatedPoint, get_screen_point_covariance(triangulatedPoint.z())));
                            const matrix33& worldPointCovariance = utils::get_world_point_covariance(vector2(triangulatedPoint.x(), triangulatedPoint.y()), triangulatedPoint.z(), get_screen_point_covariance(triangulatedPoint.z()));

                            // update this map point errors & position
                            mapPoint.update_matched(triangulatedPoint, worldPointCovariance + poseCovariance);

                            // If a new descriptor is available, update it
                            if (keypointObject.is_descriptor_computed(matchedPointIndex))
                                mapPoint._descriptor = keypointObject.get_descriptor(matchedPointIndex);
                            return;
                        }
                    }
#endif
                }
            }

            // point is unmatched
            mapPoint.update_unmatched();
        }

        void Local_Map::update_local_keypoint_map(const matrix33& poseCovariance, const matrix44& previousCameraToWorldMatrix, const matrix44& cameraToWorldMatrix, const features::keypoints::Keypoint_Handler& keypointObject)
        {
            point_map_container::iterator pointMapIterator = _localPointMap.begin();
            while(pointMapIterator != _localPointMap.end())
            {
                // Update the matched/unmatched status
                Map_Point& mapPoint = pointMapIterator->second;
                assert(pointMapIterator->first == mapPoint._id);

                update_point_match_status(mapPoint, poseCovariance, keypointObject, previousCameraToWorldMatrix, cameraToWorldMatrix);

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

        void Local_Map::update_staged_keypoints_map(const matrix33& poseCovariance, const matrix44& previousCameraToWorldMatrix, const matrix44& cameraToWorldMatrix, const features::keypoints::Keypoint_Handler& keypointObject)
        {
            // Add correct staged points to local map
            staged_point_container::iterator stagedPointIterator = _stagedPoints.begin();
            while(stagedPointIterator != _stagedPoints.end())
            {
                Staged_Point& stagedPoint = stagedPointIterator->second;
                assert(stagedPointIterator->first == stagedPoint._id);

                // Update the matched/unmatched status
                update_point_match_status(stagedPoint, poseCovariance, keypointObject, previousCameraToWorldMatrix, cameraToWorldMatrix);

                if (stagedPoint.should_add_to_local_map())
                {
                    const vector3& stagedPointCoordinates = stagedPoint._coordinates;
                    assert(not std::isnan(stagedPointCoordinates.x()) and not std::isnan(stagedPointCoordinates.y()) and not std::isnan(stagedPointCoordinates.z()));
                    // Add to local map, remove from staged points, with a copy of the id affected to the local map
                    _localPointMap.emplace(
                            stagedPoint._id,
                            Map_Point(stagedPointCoordinates, stagedPoint.get_covariance_matrix(), stagedPoint._descriptor, stagedPoint._id)
                            );
                    _localPointMap.at(stagedPoint._id)._matchedScreenPoint = stagedPoint._matchedScreenPoint;
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

        void Local_Map::add_umatched_keypoints_to_staged_map(const matrix33& poseCovariance, const matrix44& cameraToWorldMatrix, const features::keypoints::Keypoint_Handler& keypointObject)
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

                    const double depth = keypointObject.get_depth(i);
                    if (not utils::is_depth_valid(depth))
                    {
                        continue;
                    }

                    const vector2& screenPoint = keypointObject.get_keypoint(i);
                    const vector3& worldPoint = utils::screen_to_world_coordinates(screenPoint.x(), screenPoint.y(), depth, cameraToWorldMatrix);
                    assert(not std::isnan(worldPoint.x()) and not std::isnan(worldPoint.y()) and not std::isnan(worldPoint.z()));

                    const matrix33& worldPointCovariance = utils::get_world_point_covariance(screenPoint, depth, utils::get_screen_point_covariance(screenPoint, depth));

                    Staged_Point newStagedPoint(worldPoint, worldPointCovariance + poseCovariance, keypointObject.get_descriptor(i));
                    _stagedPoints.emplace(
                            newStagedPoint._id,
                            newStagedPoint);

                    MatchedScreenPoint match;
                    match._screenCoordinates << screenPoint, depth;
                    // This id is to unsure the tracking of this staged point for it's first detection
                    match._matchIndex = 0;
                    _stagedPoints.at(newStagedPoint._id)._matchedScreenPoint = match;
                }
            }

        }


        const features::keypoints::KeypointsWithIdStruct Local_Map::get_tracked_keypoints_features() const
        {
            const size_t numberOfNewKeypoints = _localPointMap.size() + _stagedPoints.size();

            // initialize output structure
            features::keypoints::KeypointsWithIdStruct keypointsWithIds; 

            // TODO: check the efficiency gain of those reserve calls
            keypointsWithIds._ids.reserve(numberOfNewKeypoints);
            keypointsWithIds._keypoints.reserve(numberOfNewKeypoints);

            // add map points with valid retroprojected coordinates
            for (const auto& [pointId, point]  : _localPointMap)
            {
                assert(pointId == point._id);
                add_point_to_tracked_features(point, keypointsWithIds);
            }
            // add staged points with valid retroprojected coordinates
            for (const auto& [pointId, point] : _stagedPoints)
            {
                assert(pointId == point._id);
                add_point_to_tracked_features(point, keypointsWithIds);
            }
            return keypointsWithIds;
        }

        void Local_Map::update_local_to_global() 
        {
            // TODO when we have a global map

        }

        void Local_Map::reset()
        {
            _localPointMap.clear();
            _stagedPoints.clear();
        }

        void Local_Map::draw_point_on_image(const IMap_Point_With_Tracking& mapPoint, const matrix44& worldToCameraMatrix, const cv::Scalar& pointColor, cv::Mat& debugImage)
        {
            if (mapPoint._matchedScreenPoint.is_matched())
            {
                vector2 screenPoint; 
                const bool isCoordinatesValid = utils::world_to_screen_coordinates(mapPoint._coordinates, worldToCameraMatrix, screenPoint);

                //Map Point are green 
                if (isCoordinatesValid)
                {
                    cv::circle(debugImage, cv::Point(screenPoint.x(), screenPoint.y()), 3, pointColor, -1);
                }
            }
        }

        void Local_Map::draw_primitives_on_image(const matrix44& worldToCameraMatrix, cv::Mat& debugImage) const
        {
            assert(not debugImage.empty());

            if (_localPrimitiveMap.size() == 0)
                return;
            const cv::Size& debugImageSize = debugImage.size();
            const uint imageWidth = debugImageSize.width;

            const double maskAlpha = 0.3;
            // 20 pixels
            const uint bandSize = 20;

            const uint placeInBand = bandSize * 0.75;
            std::stringstream text1;
            text1 << "Planes:";
            const double planeLabelPosition = imageWidth * 0.25;
            cv::putText(debugImage, text1.str(), cv::Point(planeLabelPosition, placeInBand), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 1));

            std::stringstream text2;
            text2 << "Cylinders:";
            const double cylinderLabelPosition = imageWidth * 0.60;
            cv::putText(debugImage, text2.str(), cv::Point(cylinderLabelPosition, placeInBand), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 1));

            // Tracking variables
            uint cylinderCount = 0;
            uint planeCount = 0;
            std::set<uint> alreadyDisplayedIds;

            cv::Mat allPrimitiveMasks = cv::Mat::zeros(debugImageSize, debugImage.type());
            for(const auto& [primitiveId, mapPrimitive]: _localPrimitiveMap)
            {
                // TODO find a better way to get colors
                assert(primitiveId <= _primitiveColorCodes.size());
                const cv::Scalar primitiveColor = _primitiveColorCodes[primitiveId - 1];

                cv::Mat primitiveMask;
                // Resize with no interpolation
                cv::resize(mapPrimitive._primitive->get_shape_mask() * 255, primitiveMask, debugImageSize, 0, 0, cv::INTER_NEAREST);
                cv::cvtColor(primitiveMask, primitiveMask, cv::COLOR_GRAY2BGR);
                assert(primitiveMask.size == debugImage.size);
                assert(primitiveMask.type() == debugImage.type());

                // merge with debug image
                primitiveMask.setTo(primitiveColor, primitiveMask);
                allPrimitiveMasks += primitiveMask;

                // Add color codes in label bar
                if (alreadyDisplayedIds.contains(primitiveId))
                    continue;   // already shown

                alreadyDisplayedIds.insert(primitiveId);

                double labelPosition = imageWidth;
                uint finalPlaceInBand = placeInBand;
                switch(mapPrimitive._primitive->get_primitive_type())
                {
                    case features::primitives::PrimitiveType::Plane:
                        labelPosition *= 0.25;
                        finalPlaceInBand *= planeCount;
                        ++planeCount;
                        break;
                    case features::primitives::PrimitiveType::Cylinder:
                        labelPosition *= 0.60;
                        finalPlaceInBand *= cylinderCount;
                        ++cylinderCount;
                        break;
                    default:
                        labelPosition = -1;
                        utils::log_error("Invalid enum value");
                        continue;
                }

                // make a
                const uint labelSquareSize = bandSize * 0.5;
                cv::rectangle(debugImage, 
                        cv::Point(labelPosition + 80 + finalPlaceInBand, 6),
                        cv::Point(labelPosition + 80 + labelSquareSize + finalPlaceInBand, 6 + labelSquareSize), 
                        primitiveColor,
                        -1);
            }

            cv::addWeighted(debugImage, (1 - maskAlpha), allPrimitiveMasks, maskAlpha, 0.0, debugImage);
        }

        void Local_Map::get_debug_image(const utils::Pose& camPose, const bool shouldDisplayStaged, const bool shouldDisplayPrimitiveMasks, cv::Mat& debugImage)  const
        {
            const matrix44& worldToCamMatrix = utils::compute_world_to_camera_transform(camPose.get_orientation_quaternion(), camPose.get_position());

            // Display primitives
            if (shouldDisplayPrimitiveMasks)
                draw_primitives_on_image(worldToCamMatrix, debugImage);

            // Display keypoints
            for (const auto& [pointId, mapPoint] : _localPointMap) {
                assert(pointId == mapPoint._id);
                draw_point_on_image(mapPoint, worldToCamMatrix, cv::Scalar(0, 255, 0), debugImage);
            }
            if (shouldDisplayStaged)
            {
                for (const auto& [pointId, stagedPoint] : _stagedPoints) {
                    assert(pointId == stagedPoint._id);
                    draw_point_on_image(stagedPoint, worldToCamMatrix, cv::Scalar(0, 200, 255), debugImage);
                }
            }
        }

        void Local_Map::mark_outliers_as_unmatched(const matches_containers::match_point_container& outlierMatchedPoints)
        {
            // Mark outliers as unmatched
            for (const matches_containers::Match& match : outlierMatchedPoints)
            {
                const bool isOutlierRemoved = mark_point_with_id_as_unmatched(match._mapPointId);
                // If no points were found, this is bad. A match marked as outliers must be in the local map or staged points
                assert(isOutlierRemoved == true);
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
                mark_point_with_id_as_unmatched(pointId, mapPoint);
                return true;
            }

            // Check if id is in staged points
            staged_point_container::iterator stagedPointIterator = _stagedPoints.find(pointId);
            if (stagedPointIterator != _stagedPoints.end())
            {
                Staged_Point& stagedPoint = stagedPointIterator->second;
                assert(stagedPoint._id == pointId);
                mark_point_with_id_as_unmatched(pointId, stagedPoint);
                return true;
            }

            // point associated with id was not find
            return false;
        }

        void Local_Map::mark_point_with_id_as_unmatched(const size_t pointId, IMap_Point_With_Tracking& point)
        {
            assert(pointId == point._id);
            assert(point._matchedScreenPoint._matchIndex < static_cast<int>(_isPointMatched.size()));

            // Mark point as unmatched
            _isPointMatched[point._matchedScreenPoint._matchIndex] = false;
            point._matchedScreenPoint.mark_unmatched();
        }


        void Local_Map::set_color_vector() 
        {
            for(int i = 0; i < 100; i++) {
                cv::Vec3b color;
                color[0] = rand() % 255;
                color[1] = rand() % 255;
                color[2] = rand() % 255;
                _primitiveColorCodes.push_back(color);
            }

            // Add specific colors for planes
            _primitiveColorCodes[0][0] = 0;   _primitiveColorCodes[0][1] = 0;   _primitiveColorCodes[0][2] = 255;
            _primitiveColorCodes[1][0] = 255; _primitiveColorCodes[1][1] = 0;   _primitiveColorCodes[1][2] = 204;
            _primitiveColorCodes[2][0] = 255; _primitiveColorCodes[2][1] = 100; _primitiveColorCodes[2][2] = 0;
            _primitiveColorCodes[3][0] = 0;   _primitiveColorCodes[3][1] = 153; _primitiveColorCodes[3][2] = 255;
            // Add specific colors for cylinders
            _primitiveColorCodes[50][0] = 178; _primitiveColorCodes[50][1] = 255; _primitiveColorCodes[50][2] = 0;
            _primitiveColorCodes[51][0] = 255; _primitiveColorCodes[51][1] = 0;   _primitiveColorCodes[51][2] = 51;
            _primitiveColorCodes[52][0] = 0;   _primitiveColorCodes[52][1] = 255; _primitiveColorCodes[52][2] = 51;
            _primitiveColorCodes[53][0] = 153; _primitiveColorCodes[53][1] = 0;   _primitiveColorCodes[53][2] = 255;
        }



    }   /* map_management */
}   /* rgbd_slam */
