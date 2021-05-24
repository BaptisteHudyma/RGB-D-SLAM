#ifndef IMAGE_FEATURES_HANDLER_HPP
#define IMAGE_FEATURES_HANDLER_HPP

#include "Image_Features_Struct.hpp"
#include "Parameters.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

namespace poseEstimation {

    typedef std::vector<cv::Rect> rect_vector;
    typedef std::vector<cv::KeyPoint> keypoint_vector;
    typedef std::vector<cv::Point2f> point_vector;

    struct compute_features_data
    {
        cv::Mat img;
        cv::Ptr<cv::AgastFeatureDetector> _detector;
        cv::Ptr<cv::DescriptorExtractor> _extractor;
        rect_vector _subImgsRects;
        point_vector *_ext_kp;
        Image_Features_Struct *_features_struct;
        Parameters *_voParams;
    };

    class Image_Features_Handler {
        public:
            Image_Features_Handler(const Parameters &vo_params);

            void compute_features(const cv::Mat& imgGray, const cv::Mat& imgDepth, Image_Features_Struct& outStruct);

            void row_match(Image_Features_Struct& features, Image_Features_Struct& features_right, std::vector<cv::DMatch>& outMatches);

        protected:
            void perform_compute_features(compute_features_data *);
            void perform_compute_descriptors_only(compute_features_data *);

        private:
            Parameters _voParams;
            rect_vector _subImgsRects;
            compute_features_data _thData[2];
    };

};

#endif
