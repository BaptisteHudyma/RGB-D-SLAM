#ifndef IMAGE_FEATURES_HANDLER_HPP
#define IMAGE_FEATURES_HANDLER_HPP

#include "Image_Features_Struct.hpp"
#include "Parameters.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

namespace rgbd_slam {
namespace poseEstimation {

    typedef std::vector<cv::Rect> rect_vector;
    typedef std::vector<cv::KeyPoint> keypoint_vector;
    typedef std::vector<cv::Point2f> point_vector;

    /**
     * \brief Container storing an image, feature detectors, camera parameters, image division in search regions, ...
     */
    struct compute_features_data
    {
        cv::Mat _img;
        cv::Ptr<cv::AgastFeatureDetector> _detector;
        cv::Ptr<cv::DescriptorExtractor> _extractor;
        rect_vector _subImgsRects;
        point_vector *_ext_kp;
        Image_Features_Struct *_features_struct;
        Parameters *_voParams;
    };

    /**
     * \brief Main image feature detection class. Computes image features and provide methods for feature matching
     */
    class Image_Features_Handler {
        public:
            Image_Features_Handler(const Parameters &vo_params);

            /**
             * \brief Compute 3D features from gray and depth image
             *
             * \param[in] imgGray Input image, as greyscale
             * \param[in] imgDepth Input depth image
             * \param[out] outStruct Structure of map features
             */
            void compute_features(const cv::Mat& imgGray, const cv::Mat& imgDepth, Image_Features_Struct& outStruct);

            /**
             * \brief Searches for a matches between features and our map features. Reduces search space to rows. 
             *
             * \param[in, out] featuresLeft Container of all features in the left camera image (or previous image)
             * \param[in, out] featuresRight Container of all features in the right camera image (or current frame)
             * \param[out] outMatches Container of all matches
             */
            void row_match(Image_Features_Struct& featuresLeft, Image_Features_Struct& featuresRight, std::vector<cv::DMatch>& outMatches);

        protected:
            /**
             * \brief Detect features and compute all features descriptors in an image
             */
            void perform_compute_features(compute_features_data *p);

            /**
             * \brief Compute the descriptors of a given feature set
             */
            void perform_compute_descriptors_only(compute_features_data *p);

        private:
            Parameters _voParams;
            rect_vector _subImgsRects;
            compute_features_data _thData[2];
    };

}
}

#endif
