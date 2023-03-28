#ifndef RGBDSLAM_FEATURES_PRIMITIVES_DEPTHOPERATIONS_HPP
#define RGBDSLAM_FEATURES_PRIMITIVES_DEPTHOPERATIONS_HPP

#include "../../types.hpp"
#include <opencv2/opencv.hpp>

namespace rgbd_slam::features::primitives {

/**
 * \brief Handles operations on the initial depth image, to transform it on a connected cloud points.
 * It also handles the loading of the camera parameters from the configuration file
 */
class Depth_Map_Transformation
{
  public:
    /**
     * \param[in] width Depth image width (constant)
     * \param[in] height Depth image height (constant)
     * \param[in] cellSize Size of the cloud point division (> 0)
     */
    Depth_Map_Transformation(const uint width, const uint height, const uint cellSize);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * \brief Rectify the given depth image to align it with the RGB image
     * \param[in] depthImage The unrectified depth image
     * \param[in] rectifiedDepth The depth image, transformed to align with the rgb image
     * \return True if the transformation was successful
     */
    bool rectify_depth(const cv::Mat& depthImage, cv::Mat& rectifiedDepth);

    /**
     * \brief Create an point cloud organized by cells of cellSize*cellSize pixels
     *
     * \param[in] depthImage Input depth image representation, transformed to align to rgb image at output
     * \param[out] organizedCloudArray A cloud point divided in blocs of cellSize * cellSize
     * \return True if the process succeeded
     */
    bool get_organized_cloud_array(const cv::Mat& depthImage, matrixf& organizedCloudArray);

    /**
     * \brief Controls the state of this class.
     *
     * \return False if the camera parameters could not be loaded
     */
    bool is_ok() const { return _isOk; };

  protected:
    /**
     * \brief Loads the camera intrinsic parameters
     */
    bool load_parameters();

    /**
     * \brief Must be called after load_parameters. Fills the computation matrices
     */
    void init_matrices();

  private:
    uint _width;
    uint _height;
    uint _cellSize;
    bool _isOk;

    // cam parameters
    double _fxIr;
    double _fyIr;
    double _cxIr;
    double _cyIr;

    double _fxRgb;
    double _fyRgb;
    double _cxRgb;
    double _cyRgb;

    // camera parameters
    cv::Mat _Rstereo;
    cv::Mat _Tstereo;

    // pre computation matrix
    cv::Mat_<float> _Xpre;
    cv::Mat_<float> _Ypre;
    cv::Mat_<int> _cellMap;
};

} // namespace rgbd_slam::features::primitives

#endif
