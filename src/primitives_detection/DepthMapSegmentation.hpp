#ifndef DEPTH_MAP_SEGMENTATION
#define DEPTH_MAP_SEGMENTATION

#include <opencv2/opencv.hpp>

namespace rgbd_slam {
namespace primitiveDetection {

    /**
     * \brief Compute the normal map associated with a depth image
     *
     * \param[in] depthMap Depth map with format CV_32FC1
     * \param[out] normalMap The computed normal map
     */
    void get_normal_map(const cv::Mat& depthMap, cv::Mat& normalMap) ;

    /**
     * \brief Compute an edge map from a normal map
     *
     * \param[in] depthMap Depth map with format CV_32FC1
     * \param[in] normalMap Normal map extracted from depthMap
     * \param[out] edgeMap Edge map, computed from a combinaison of a convex and a discontinuity operator
     */
    void get_edge_masks(const cv::Mat& depthMap, const cv::Mat& normalMap, cv::Mat& edgeMap);

    /**
      * \brief Return the depth map segmented as individual objects
      *
      * \param[in] depthMap Depth map with normal format CV_32FC1
      * \param[out] finalSegmented Final segmented image, where each pixel is associated with an ID
      * \param[in] kernel The kernel to use for morphological operations
      * \param[in] reducePourcent Between 0 and 1, coefficient of reduction of the initial depth image. Lower the resolution but improves performances
      */
    void get_segmented_depth_map(const cv::Mat& depthMap, cv::Mat& finalSegmented, const cv::Mat& kernel, double reducePourcent=0.5);

    /**
      * \brief Draw individual segmented labels on an image
      *
      * \param[in] segmentedImage Segmented image, where each pixel is associated with a shape ID
      * \param[in] colors Vector containing a set of colors
      * \param[out] outputImage Final segmented image
      */
    void draw_segmented_labels(const cv::Mat& segmentedImage, const std::vector<cv::Vec3b>& colors, cv::Mat& outputImage);
}
}

#endif
