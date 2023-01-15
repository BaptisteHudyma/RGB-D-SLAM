#ifndef RGBDSLAM_FEATURES_KEYPOINTS_KEYPOINTS_HANDLER_HPP
#define RGBDSLAM_FEATURES_KEYPOINTS_KEYPOINTS_HANDLER_HPP

#include <list>
#include <opencv2/core/types.hpp>
#include <utility>
#include <vector>
#include <opencv2/xfeatures2d.hpp>

#include "../../utils/coordinates.hpp"

namespace rgbd_slam {
    namespace features {
        namespace keypoints {

            const int BORDER_SIZE = 1; // Border of an image, in which points will be ignored
            const size_t INVALID_MAP_POINT_ID = 0;  // should be the same as INVALID_POINT_UNIQ_ID in map_point.hpp
            const int INVALID_MATCH_INDEX = -1;

            struct DetectedKeyPoint
            {
                utils::ScreenCoordinate _coordinates;
                cv::Mat _descriptor;

                bool can_add_to_map() const
                {
                    // descriptor is not empty, depth is valid
                    return (not _descriptor.empty()) and (utils::is_depth_valid(_coordinates.z()));
                }
            };


            /**
             * \brief checks if a point is in an image, a with border
             */
            bool is_in_border(const cv::Point2f &pt, const cv::Mat &im, const double borderSize = 0);

            /**
              * \brief Return the depth value in the depth image, or 0 if not depth info is found. This function approximates depth with the surrounding points to prevent invalid depth on edges
              */
            double get_depth_approximation(const cv::Mat& depthImage, const cv::Point2f& depthCoordinates);

            /**
             * \brief Stores a vector of keypoints, along with a vector of the unique ids associated with those keypoints in the local map
             */
            struct KeypointsWithIdStruct {
                struct keypointWithId {
                    size_t _id;
                    cv::Point2f _point;

                    keypointWithId(const size_t id, const cv::Point2f& point):
                    _id(id), _point(point)
                    {};
                };

                void clear()
                {
                    _keypoints.clear();
                    _ids.clear();
                }

                void reserve(const size_t numberOfNewKeypoints)
                {
                    _keypoints.reserve(numberOfNewKeypoints);
                    _ids.reserve(numberOfNewKeypoints);
                }

                size_t size() const
                {
                    assert(_keypoints.size() == _ids.size());
                    return _keypoints.size();
                }

                bool empty() const
                {
                    return size() == 0;
                }

                const keypointWithId at(const size_t index) const
                {
                    assert(index < size());
                    return keypointWithId(_ids[index], _keypoints[index]);
                }

                void add(const size_t id, const double pointX, const double pointY)
                {
                    add(id, 
                        cv::Point2f(static_cast<float>(pointX), static_cast<float>(pointY))
                    );
                }
                void add(const size_t id, const cv::Point2f point)
                {
                    _keypoints.emplace_back(point);
                    _ids.emplace_back(id);
                }

                const std::vector<cv::Point2f>& get_keypoints() const
                {
                    return _keypoints;
                }
                const std::vector<size_t>& get_ids() const
                {
                    return _ids;
                }

                private:
                std::vector<cv::Point2f> _keypoints;
                std::vector<size_t> _ids;
            };

            /**
             * \brief Handler object to store a reference to detected key points. Passed to other classes for data association purposes
             */
            class Keypoint_Handler
            {
                public:
                    /**
                     * \param[in] maxMatchDistance Maximum distance to consider that a match of two points is valid
                     */
                    Keypoint_Handler(const uint depthImageCols, const uint depthImageRows, const double maxMatchDistance = 0.7);

                    /**
                     * \brief Set the container properties
                     * \param[in] inKeypoints New keypoints detected, no tracking informations
                     * \param[in] inDescriptors Descriptors of the new keypoints
                     * \param[in] lastKeypointsWithIds Keypoints tracked with optical flow, and their matching ids
                     * \param[in] depthImage The depth image in which those keypoints were detected
                     */
                    void set(std::vector<cv::Point2f>& inKeypoints, cv::Mat& inDescriptors, const KeypointsWithIdStruct& lastKeypointsWithIds, const cv::Mat& depthImage);

                    /**
                     * \brief Get a tracking index if it exist, or -1.
                     *
                     * \param[in] mapPointId The unique id that we want to check
                     * \param[in] isKeyPointMatchedContainer A vector of size _keypoints, use to flag is a keypoint is already matched 
                     *
                     * \return the index of the tracked point in _keypoints, or -1 if no match was found
                     */
                    int get_tracking_match_index(const size_t mapPointId, const vectorb& isKeyPointMatchedContainer) const;
                    int get_tracking_match_index(const size_t mapPointId) const;

                    /**
                     * \brief get an index corresponding to the index of the point matches.
                     *
                     * \param[in] projectedMapPoint A 2D map point to match 
                     * \param[in] mapPointDescriptor The descriptor of this map point
                     * \param[in] isKeyPointMatchedContainer A vector of size _keypoints, use to flag is a keypoint is already matched 
                     *
                     * \return An index >= 0 corresponding to the matched keypoint, or -1 if no match was found
                     */
                    int get_match_index(const utils::ScreenCoordinate2D& projectedMapPoint, const cv::Mat& mapPointDescriptor, const vectorb& isKeyPointMatchedContainer, const double searchSpaceRadius) const; 

                    /**
                     * \brief return the keypoint associated with the index
                     */
                    const utils::ScreenCoordinate get_keypoint(const uint index) const 
                    {
                        assert(index < _keypoints.size());
                        return _keypoints[index];
                    }

                    bool is_descriptor_computed(const uint index) const
                    {
                        return index < static_cast<uint>(_descriptors.rows);
                    }

                    const cv::Mat get_descriptor(const uint index) const
                    {
                        assert(index < static_cast<uint>(_descriptors.rows));

                        return _descriptors.row(index);
                    }

                    size_t get_keypoint_count() const
                    {
                        return _keypoints.size();
                    }

                    size_t size() const { return _keypoints.size(); };

                    DetectedKeyPoint at(const size_t index) const 
                    {
                        DetectedKeyPoint newKp;
                        newKp._coordinates = get_keypoint(index);
                        if(is_descriptor_computed(index))
                            newKp._descriptor = get_descriptor(index);
                        else
                            newKp._descriptor = cv::Mat();
                        return newKp;
                    }

                protected:

                    /**
                     * \brief Return a mask eliminating the keypoints to far from the point to match
                     *
                     * \param[in] pointToSearch The 2D screen coordinates of the point to match
                     * \param[in] isKeyPointMatchedContainer A vector of size _keypoints, use to flag is a keypoint is already matched 
                     * \param[in] searchSpaceCellRadius Radius of the match search
                     *
                     * \return A Mat the same size as our keypoint array, with 0 where the index is not a candidate, and 1 where it is
                     */
                    const cv::Mat compute_key_point_mask(const utils::ScreenCoordinate2D& pointToSearch, const vectorb& isKeyPointMatchedContainer, const uint searchSpaceCellRadius) const;

                    typedef std::pair<uint, uint> uint_pair;
                    /**
                     * \brief Returns a 2D id corresponding to the X and Y of the search space in the image. The search space indexes must be used with _searchSpaceIndexContainer
                     */
                    const uint_pair get_search_space_coordinates(const utils::ScreenCoordinate2D& pointToPlace) const;

                    /**
                     * \brief Compute an 1D array index from a 2D array index.
                     *
                     * \param[in] searchSpaceIndex The 2D array index (y, x)
                     */
                    uint get_search_space_index(const uint_pair& searchSpaceIndex) const;
                    uint get_search_space_index(const uint x, const uint y) const;

                    void clear();


                private:
                    cv::Ptr<cv::DescriptorMatcher> _featuresMatcher;

                    const double _maxMatchDistance;

                    //store current frame keypoints
                    std::vector<utils::ScreenCoordinate> _keypoints;
                    typedef std::unordered_map<size_t, size_t> uintToUintContainer;
                    uintToUintContainer _uniqueIdsToKeypointIndex;
                    cv::Mat _descriptors;

                    // Number of image divisions (cells)
                    uint _cellCountX;
                    uint _cellCountY;

                    // Corresponds to a 2D box containing index of key points in those boxes
                    typedef std::vector<uint> index_container;
                    std::vector<index_container> _searchSpaceIndexContainer;
            };

        }
    }
}

#endif
