#ifndef KEY_POINT_EXTRATION_HPP
#define KEY_POINT_EXTRATION_HPP

#include <opencv2/xfeatures2d.hpp>
#include <map>
#include <list>

#include "types.hpp"
#include "map_point.hpp"
#include "PoseUtils.hpp"

namespace rgbd_slam {
    namespace utils {

        class Keypoint_Handler
        {
            public:
                /**
                 * \param[in] maxMatchDistance Maximum distance to consider that a match of two points is valid
                 */
                Keypoint_Handler(std::vector<cv::KeyPoint>& inKeypoints, cv::Mat& inDescriptors, const cv::Mat& depthImage, const double maxMatchDistance = 0.7);

                /**
                 * \brief get a container with the filtered point matches. Uses _maxMatchDistance to estimate good matches. 
                 *
                 * \param[in] mapPoint A point to match 
                 *
                 * \return An index >= 0 corresponding to the matched keypoint, or -1 if no match was found
                 */
                int get_match_index(const map_management::Point& mapPoint) const; 

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

                    const cv::Point2f point = _keypoints[index].pt;
                    vector2 keypoint;
                    keypoint << point.x, point.y;
                    return keypoint;
                }

                const cv::Mat get_descriptor(const unsigned int index) const
                {
                    assert(index < _keypoints.size());

                    return _descriptors.row(index);
                }

                unsigned int get_keypoint_count() const
                {
                    return _keypoints.size();
                }


            private:
                cv::Ptr<cv::DescriptorMatcher> _featuresMatcher;

                const double _maxMatchDistance;

                //store current frame keypoints
                std::vector<cv::KeyPoint> _keypoints;
                std::vector<double> _depths;
                cv::Mat _descriptors;


        };

        /**
         * \brief A class to detect and store keypoints
         *
         */
        class Key_Point_Extraction 
        {
            public:

                /**
                 *
                 */
                Key_Point_Extraction(const unsigned int minHessian = 25);

                /**
                 * \brief detect the keypoints in the gray image 
                 *
                 * \param[in] grayImage The input image from camera
                 *
                 * \return An object that contains the detected keypoints
                 */
                const Keypoint_Handler detect_keypoints(const cv::Mat& grayImage, const cv::Mat& depthImage);


                /**
                 * \brief Show the time statistics for certain parts of the program. Kind of a basic profiler
                 *
                 * \param[in] meanFrameTreatmentTime The mean time in seconds that this program used to treat one frame
                 * \param[in] frameCount Total of frame treated by the program
                 */
                void show_statistics(const double meanFrameTreatmentTime, const unsigned int frameCount) const;


            private:
                cv::Ptr<cv::FeatureDetector> _featureDetector;
                cv::Ptr<cv::DescriptorExtractor> _descriptorExtractor;

                double _meanPointExtractionTime;

        };

    }
}

#endif
