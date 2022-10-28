#ifndef RGBDSLAM_FEATURES_KEYPOINTS_KEYPOINTS_HANDLER_HPP
#define RGBDSLAM_FEATURES_KEYPOINTS_KEYPOINTS_HANDLER_HPP

#include <list>
#include <vector>
#include <opencv2/xfeatures2d.hpp>

#include "../../utils/coordinates.hpp"

namespace rgbd_slam {
    namespace features {
        namespace keypoints {

            const double BORDER_SIZE = 1.; // Border of an image, in which points will be ignored
            const size_t INVALID_MAP_POINT_ID = 0;  // should be the same as INVALID_POINT_UNIQ_ID in map_point.hpp
            const int INVALID_MATCH_INDEX = -1;

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
                     * \param[in] inKeypoints New keypoints detected, no tracking informations
                     * \param[in] inDescriptors Descriptors of the new keypoints
                     * \param[in] lastKeypointsWithIds Keypoints tracked with optical flow, and their matching ids
                     * \param[in] depthImage The depth image in which those keypoints were detected
                     * \param[in] maxMatchDistance Maximum distance to consider that a match of two points is valid
                     */
                    Keypoint_Handler(std::vector<cv::Point2f>& inKeypoints, cv::Mat& inDescriptors, const KeypointsWithIdStruct& lastKeypointsWithIds, const cv::Mat& depthImage, const double maxMatchDistance = 0.7);

                    /**
                     * \brief Get a tracking index if it exist, or -1.
                     *
                     * \param[in] mapPointId The unique id that we want to check
                     * \param[in] isKeyPointMatchedContainer A vector of size _keypoints, use to flag is a keypoint is already matched 
                     *
                     * \return the index of the tracked point in _keypoints, or -1 if no match was found
                     */
                    int get_tracking_match_index(const size_t mapPointId, const std::vector<bool>& isKeyPointMatchedContainer) const;
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
                    int get_match_index(const utils::ScreenCoordinate2D& projectedMapPoint, const cv::Mat& mapPointDescriptor, const std::vector<bool>& isKeyPointMatchedContainer) const; 

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

                protected:

                    /**
                     * \brief Return a mask eliminating the keypoints to far from the point to match
                     *
                     * \param[in] pointToSearch The 2D screen coordinates of the point to match
                     * \param[in] isKeyPointMatchedContainer A vector of size _keypoints, use to flag is a keypoint is already matched 
                     *
                     * \return A Mat the same size as our keypoint array, with 0 where the index is not a candidate, and 1 where it is
                     */
                    const cv::Mat compute_key_point_mask(const utils::ScreenCoordinate2D& pointToSearch, const std::vector<bool>& isKeyPointMatchedContainer) const;

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
                    uint _searchSpaceCellRadius; 

                    // Corresponds to a 2D box containing index of key points in those boxes
                    typedef std::list<uint> index_container;
                    std::vector<index_container> _searchSpaceIndexContainer;


            };

        }
    }
}

#endif
