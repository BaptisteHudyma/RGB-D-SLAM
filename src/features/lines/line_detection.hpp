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

                    void detect_lines(const cv::Mat& grayImage, const cv::Mat& depthImage, cv::Mat& outImage);

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
