#ifndef DEPTH_MAP_SEGMENTATION
#define DEPTH_MAP_SEGMENTATION

#include <opencv2/opencv.hpp>

namespace primitiveDetection {

    void get_normal_map(const cv::Mat& depthMap, cv::Mat& normalMap) ;

    void get_edge_masks(const cv::Mat& depthMap, const cv::Mat& normalMap, cv::Mat& edgeMap);

    void get_segmented_depth_map(const cv::Mat& depthMap, cv::Mat& finalSegmented, const cv::Mat& kernel, double reducePourcent=0.5);

    void draw_segmented_labels(const cv::Mat& segmentedImage, const std::vector<cv::Vec3b>& colors, cv::Mat& outputImage);
}

#endif
