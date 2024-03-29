#include "line_detection.hpp"
#include "../../outputs/logger.hpp"
#include <memory>

namespace rgbd_slam::features::lines {

Line_Detection::Line_Detection(const double scale, const double sigmaScale)
{
    assert(scale > 0 and scale <= 1);
    assert(sigmaScale > 0 and sigmaScale <= 1);

    // Should refine, scale, Gaussian filter sigma
    _lineDetector = std::make_unique<cv::LSD>(cv::LSD_REFINE_NONE, scale, sigmaScale);

    if (_lineDetector == nullptr)
    {
        outputs::log_error("Instanciation of LSD failed");
        exit(-1);
    }

    _kernel = cv::Mat_<uchar>::ones(3, 3);
}

line_container Line_Detection::detect_lines(const cv::Mat& grayImage, const cv::Mat_<float>& depthImage) noexcept
{
    assert(_lineDetector != nullptr);

    // get lines
    line_container rawLines;
    _lineDetector->detect(grayImage, rawLines);
    return rawLines;
}

void Line_Detection::get_image_with_lines(const line_container& linesToDisplay,
                                          const cv::Mat_<float>& depthImage,
                                          cv::Mat& outImage) const noexcept
{
    // draw lines with associated depth data
    if (linesToDisplay.empty())
    {
        return;
    }
    // binarize depth map, fill holes
    const cv::Mat_<uchar> mask = depthImage > 0;
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, _kernel);

    for (const cv::Vec4f& pts: linesToDisplay)
    {
        const cv::Point pt1(static_cast<int>(pts[0]), static_cast<int>(pts[1]));
        const cv::Point pt2(static_cast<int>(pts[2]), static_cast<int>(pts[3]));

        if (mask.at<uchar>(pt1) == 0 or mask.at<uchar>(pt2) == 0)
        {
            // no depth at extreme points, check first and second quarter
            const cv::Point firstQuart = 0.25 * pt1 + 0.75 * pt2;
            const cv::Point secQuart = 0.75 * pt1 + 0.25 * pt2;

            // at least a point with depth data
            if (mask.at<uchar>(firstQuart) != 0 or mask.at<uchar>(secQuart) != 0)
                cv::line(outImage, pt1, pt2, cv::Scalar(0, 0, 255), 1);
            else // no depth data
                cv::line(outImage, pt1, pt2, cv::Scalar(255, 0, 255), 1);
        }
        else
        {
            // line with associated depth
            cv::line(outImage, pt1, pt2, cv::Scalar(0, 255, 255), 1);
        }
    }
}

void Line_Detection::show_statistics(const double meanFrameTreatmentDuration, const uint frameCount) const noexcept
{
    static auto get_percent_of_elapsed_time = [](double treatmentTime, double totalTimeElapsed) {
        if (totalTimeElapsed <= 0)
            return 0.0;
        return (treatmentTime / totalTimeElapsed) * 100.0;
    };

    if (frameCount > 0)
    {
        const double meanLineExtractionDuration = _meanLineTreatmentDuration / static_cast<double>(frameCount);
        outputs::log(std::format("\tMean line extraction time is {:.4f} seconds ({:.2f}%)",
                                 meanLineExtractionDuration,
                                 get_percent_of_elapsed_time(meanLineExtractionDuration, meanFrameTreatmentDuration)));
    }
}

} // namespace rgbd_slam::features::lines