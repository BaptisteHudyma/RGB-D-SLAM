#ifndef RGBDSLAM_FEATURES_KEYPOINTS_KEYPOINTS_HPP
#define RGBDSLAM_FEATURES_KEYPOINTS_KEYPOINTS_HPP

#include <opencv2/xfeatures2d.hpp>
#include <map>
#include <list>

#include "types.hpp"
#include "map_point.hpp"

namespace rgbd_slam {
    namespace features {
        namespace keypoints {

            const size_t INVALID_MAP_POINT_ID = 0;  // should be the same as INVALID_POINT_UNIQ_ID in map_point.hpp
            const int INVALID_MATCH_INDEX = -1;

            /**
             * \brief Stores a vector of keypoints, along with a vector of the unique ids associated with those keypoints in the local map
             */
            struct KeypointsWithIdStruct {
                std::vector<cv::Point2f> _keypoints;
                std::vector<size_t> _ids;
            };

            /**
             * \brief Handler object to store a reference to detected key points. Passed to classes like Local_Map for data association
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
                    int get_match_index(const vector2& projectedMapPoint, const cv::Mat& mapPointDescriptor, const std::vector<bool>& isKeyPointMatchedContainer) const; 

                    /**
                     * \brief Return the depth associated with a certain keypoint
                     */
                    double get_depth(const unsigned int index) const
                    {
                        assert(index < _depths.size());

                        return _depths[index];
                    }

                    double get_depth_count() const
                    {
                        return _depths.size();
                    }

                    /**
                     * \brief return the keypoint associated with the index
                     */
                    const vector2 get_keypoint(const unsigned int index) const 
                    {
                        assert(index < _keypoints.size());
                        return _keypoints[index];
                    }

                    bool is_descriptor_computed(const unsigned int index) const
                    {
                        return index < static_cast<unsigned int>(_descriptors.rows);
                    }

                    const cv::Mat get_descriptor(const unsigned int index) const
                    {
                        assert(index < static_cast<unsigned int>(_descriptors.rows));

                        return _descriptors.row(index);
                    }

                    unsigned int get_keypoint_count() const
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
                    const cv::Mat compute_key_point_mask(const vector2& pointToSearch, const std::vector<bool>& isKeyPointMatchedContainer) const;

                    typedef std::pair<int, int> int_pair;
                    /**
                     * \brief Returns a 2D id corresponding to the X and Y of the search space in the image. The search space indexes must be used with _searchSpaceIndexContainer
                     */
                    const int_pair get_search_space_coordinates(const vector2& pointToPlace) const;

                    /**
                     * \brief Compute an 1D array index from a 2D array index.
                     *
                     * \param[in] searchSpaceIndex The 2D array index (y, x)
                     */
                    unsigned int get_search_space_index(const int_pair& searchSpaceIndex) const;
                    unsigned int get_search_space_index(const unsigned int x, const unsigned int y) const;


                private:
                    cv::Ptr<cv::DescriptorMatcher> _featuresMatcher;

                    const double _maxMatchDistance;

                    //store current frame keypoints
                    std::vector<vector2> _keypoints;
                    std::vector<double> _depths;
                    typedef std::unordered_map<size_t, size_t> intToIntContainer;
                    intToIntContainer _uniqueIdsToKeypointIndex;
                    cv::Mat _descriptors;

                    // Number of image divisions (cells)
                    int _cellCountX;
                    int _cellCountY;
                    int _searchSpaceCellRadius; 

                    // Corresponds to a 2D box containing index of key points in those boxes
                    typedef std::list<unsigned int> index_container;
                    std::vector<index_container> _searchSpaceIndexContainer;


            };



            /**
             * \brief A class to detect and store keypoints
             */
            class Key_Point_Extraction 
            {
                public:

                    /**
                     *
                     */
                    Key_Point_Extraction(const unsigned int minHessian = 25);

                    /**
                     * \brief compute the keypoints in the gray image, using optical flow and/or generic feature detectors 
                     *
                     * \param[in] grayImage The input image from camera
                     * \param[in] depthImage The input depth image from camera
                     * \param[in] lastKeypointsWithIds The keypoints of the previous detection step, that will be tracked with optical flow
                     * \param[in] forceKeypointDetection Force the detection of keypoints in the image
                     *
                     * \return An object that contains the detected keypoints
                     */
                    const Keypoint_Handler compute_keypoints(const cv::Mat& grayImage, const cv::Mat& depthImage, const KeypointsWithIdStruct& lastKeypointsWithIds, const bool forceKeypointDetection = false);


                    /**
                     * \brief Show the time statistics for certain parts of the program. Kind of a basic profiler
                     *
                     * \param[in] meanFrameTreatmentTime The mean time in seconds that this program used to treat one frame
                     * \param[in] frameCount Total of frame treated by the program
                     */
                    void show_statistics(const double meanFrameTreatmentTime, const unsigned int frameCount) const;

                protected:

                    /**
                     * \brief Compute the current frame keypoints from optical flow. There is need for matching with this configuration
                     *
                     * \param[in] imagePreviousPyramide The pyramid representation of the previous image
                     * \param[in] imageCurrentPyramide The pyramid representation of the current image to analyze
                     * \param[in] lastKeypointsWithIds The keypoints detected in imagePrevious
                     * \param[in] pyramidDepth The chosen depth of the image pyramids
                     * \param[in] windowSize The chosen size of the optical flow window 
                     * \param[in] errorThreshold an error Threshold for optical flow, in pixels
                     * \param[in] maxDistanceThreshold a distance threshold, in pixels
                     */
                    KeypointsWithIdStruct get_keypoints_from_optical_flow(const std::vector<cv::Mat>& imagePreviousPyramide, const std::vector<cv::Mat>& imageCurrentPyramide, const KeypointsWithIdStruct& lastKeypointsWithIds, const uint pyramidDepth, const uint windowSize, const double errorThreshold, const double maxDistanceThreshold) const;


                    /**
                     * \brief Compute new key point, with an optional mask to exclude detection zones
                     *
                     * \param[in] grayImage The image in which we want to detect waypoints
                     * \param[in] mask The mask which we do not want to detect waypoints
                     * \param[in] minimumPointsForValidity The minimum number of points under which we will use the precise detector
                     *
                     * \return An array of points in the input image
                     */
                    const std::vector<cv::Point2f> detect_keypoints(const cv::Mat& grayImage, const cv::Mat& mask, const uint minimumPointsForValidity) const;

                    const cv::Mat compute_key_point_mask(const cv::Size imageSize, const std::vector<cv::Point2f> keypointContainer) const;

                private:
                    cv::Ptr<cv::FeatureDetector> _featureDetector;
                    cv::Ptr<cv::FeatureDetector> _advancedFeatureDetector;
                    cv::Ptr<cv::DescriptorExtractor> _descriptorExtractor;

                    std::vector<cv::Mat> _lastFramePyramide;

                    double _meanPointExtractionTime;

            };

        }
    }
}

#endif
