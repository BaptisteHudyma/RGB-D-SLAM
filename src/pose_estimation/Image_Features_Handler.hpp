#ifndef IMAGE_FEATURES_HANDLER_HPP
#define IMAGE_FEATURES_HANDLER_HPP

#include "Image_Features_Struct.hpp"
#include "Parameters.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

namespace poseEstimation {

    struct compute_features_data
    {
        cv::Mat img;
        cv::Ptr<cv::AgastFeatureDetector> detector;
        cv::Ptr<cv::DescriptorExtractor> extractor;
        std::vector<cv::Rect> subImgsRects;
        std::vector<cv::Point2f> *ext_kp;
        Image_Features_Struct *features_struct;
        Parameters *voParams;
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
            Parameters voParams;
            std::vector<cv::Rect> subImgsRects;
            compute_features_data thData[2];
    };

};

#endif
