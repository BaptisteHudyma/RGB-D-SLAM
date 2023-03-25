#include "line_detection.hpp"
#include "../../outputs/logger.hpp"

namespace rgbd_slam {
namespace features {
namespace lines {

Line_Detection::Line_Detection(const double scale, const double sigmaScale)
{
    assert(scale > 0 and scale <= 1);
    assert(sigmaScale > 0 and sigmaScale <= 1);

    // Should refine, scale, Gaussian filter sigma
    _lineDetector = new cv::LSD(cv::LSD_REFINE_NONE, scale, sigmaScale);

    if (_lineDetector == nullptr)
    {
        outputs::log_error("Instanciation of LSD failed");
        exit(-1);
    }

    _kernel = cv::Mat::ones(3, 3, CV_8U);
}

line_container Line_Detection::detect_lines(const cv::Mat& grayImage, const cv::Mat& depthImage)
{
    // get lines
    line_container rawLines;
    _lineDetector->detect(grayImage, rawLines);
    return rawLines;
}

void Line_Detection::get_image_with_lines(const line_container& linesToDisplay,
                                          const cv::Mat& depthImage,
                                          cv::Mat& outImage)
{
    // draw lines with associated depth data
    if (not linesToDisplay.empty())
    {
        // binarize depth map, fill holes
        const cv::Mat mask = depthImage > 0;
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, _kernel);

        for (const cv::Vec4f& pts: linesToDisplay)
        {
            const cv::Point pt1(pts[0], pts[1]);
            const cv::Point pt2(pts[2], pts[3]);

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
                // line with associated depth
                cv::line(outImage, pt1, pt2, cv::Scalar(0, 255, 255), 1);
        }
    }
}

} // namespace lines
} // namespace features
} // namespace rgbd_slam