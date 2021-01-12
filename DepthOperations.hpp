#ifndef DEPTH_OPERATIONS_H
#define DEPTH_OPERATIONS_H 

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace planeDetection {

    class Depth_Operations {
        public:
            Depth_Operations(const std::string& paramFilePath, int width, int height, int cellSize);

            void get_organized_cloud_array(cv::Mat& depthImage, Eigen::MatrixXf& organizedCloudArray);
            bool is_ok() const {return isOk;};

        protected:
            bool load_parameters(const std::string& parameterFilePath);
            void init_matrices();

        private:
            int width;
            int height;
            int cellSize;
            bool isOk;
            
            Eigen::MatrixXf cloudArray;

            //cam parameters
            float fxIr;
            float fyIr;
            float cxIr;
            float cyIr;

            float fxRgb;
            float fyRgb;
            float cxRgb;
            float cyRgb;

            //camera parameters
            cv::Mat Krgb;
            cv::Mat Kir;
            cv::Mat Rstereo;
            cv::Mat Tstereo;
        
            //pre computation matrix
            cv::Mat_<float> X;
            cv::Mat_<float> Y;
            cv::Mat_<float> Xt;
            cv::Mat_<float> Yt;
            cv::Mat_<float> Xpre;
            cv::Mat_<float> Ypre;
            cv::Mat_<float> U;   
            cv::Mat_<float> V;
            cv::Mat_<int> cellMap;
    };
}

#endif
