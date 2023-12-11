#ifndef RGBDSLAM_FEATURES_LINES_LINE_DETECTION_HPP
#define RGBDSLAM_FEATURES_LINES_LINE_DETECTION_HPP

#include "../../../third_party/line_segment_detector.hpp"
#include <memory>
#include <opencv2/opencv.hpp>

namespace rgbd_slam::features::lines {

using line_container = std::vector<cv::Vec4f>;

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
    [[nodiscard]] line_container detect_lines(const cv::Mat& grayImage, const cv::Mat_<float>& depthImage) noexcept;

    /**
     * \brief display the given lines on an image
     *
     * \param[in] linesToDisplay Container of lines that will be displayed
     * \param[in] depthImage The measured depth of the image in which the lines were detected
     * \param[in, out] outImage The image on which to display the lines
     */
    void get_image_with_lines(const line_container& linesToDisplay,
                              const cv::Mat_<float>& depthImage,
                              cv::Mat& outImage) const noexcept;

    void show_statistics(const double meanFrameTreatmentDuration, const uint frameCount) const noexcept;

    double _meanLineTreatmentDuration = 0.0;

  private:
    // LineSegmentDetector
    std::unique_ptr<cv::LSD> _lineDetector;

    // kernel for morphological operations
    cv::Mat_<uchar> _kernel;

    // remove copy constructors as we have dynamically instantiated members
    Line_Detection(const Line_Detection& lineDetector) = delete;
    Line_Detection& operator=(const Line_Detection& lineDetector) = delete;
};

} // namespace rgbd_slam::features::lines

#endif
