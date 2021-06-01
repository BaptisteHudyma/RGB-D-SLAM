#ifndef MONOCULAR_DEPTH_MAP
#define MONOCULAR_DEPTH_MAP

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>

/**
  * \brief Depth map estimation based on monocular RGB image tracking
  */
class Monocular_Depth_Map {
    public:
        Monocular_Depth_Map(cv::Mat& firstImage);

        void get_monocular_depth(cv::Mat& image, cv::Mat& depthMap);

    private:
        cv::Ptr<cv::xfeatures2d::SURF> featureDetector;
        cv::Ptr<cv::DescriptorMatcher> featuresMatcher;
        cv::Ptr<cv::StereoSGBM> stereoDepthmapCompute;
        cv::Ptr<cv::ximgproc::DisparityWLSFilter> stereoFilter;

        int indexOfLast;
        std::vector<cv::KeyPoint> keypoints[2];
        cv::Mat descriptors[2];
        cv::Mat lastUndistorededImage;
};


#endif

