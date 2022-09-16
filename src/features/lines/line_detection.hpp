#ifndef RGBDSLAM_FEATURES_LINES_LINE_DETECTION_HPP
#define RGBDSLAM_FEATURES_LINES_LINE_DETECTION_HPP

#include <opencv2/opencv.hpp>
#include "../../third_party/line_segment_detector.hpp"

namespace rgbd_slam {
    namespace features {
        namespace lines {
            
            typedef std::vector<cv::Vec4f> line_container;

            /**
             * \brief A class to detect and store lines
             */
            class Line_Detection 
            {
                public:
                    Line_Detection(const double scale = 1.0, const double sigmaScale = 1.0);

                    /**
                     * \brief Search for lines in an gray image
                     *
                     * \param[in] grayImage The image in which to detect the lines
                     * \param[in] depthImage The depth dimention of the image
                     *
                     * \return A container with the detected lines (2D start and end point)
                     */
                    line_container detect_lines(const cv::Mat& grayImage, const cv::Mat& depthImage);

                    /**
                     * \brief display the given lines on an image
                     *
                     * \param[in] linesToDisplay Container of lines that will be displayed
                     * \param[in] depthImage The measured depth of the image in which the lines were detected
                     * \param[in, out] outImage The image on which to display the lines 
                     */
                    void get_image_with_lines(const line_container& linesToDisplay, const cv::Mat& depthImage, cv::Mat& outImage);

                private:
                    // LineSegmentDetector
                    cv::LSD* _lineDetector;

                    // kernel for morphological operations
                    cv::Mat _kernel;

                    // remove copy constructors as we have dynamically instantiated members
                    Line_Detection(const Line_Detection& lineDetector) = delete;
                    Line_Detection& operator=(const Line_Detection& lineDetector) = delete;
            };

        }
    }
}



#endif
