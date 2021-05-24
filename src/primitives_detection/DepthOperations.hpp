#ifndef DEPTH_OPERATIONS_H
#define DEPTH_OPERATIONS_H 

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace primitiveDetection {

    class Depth_Operations {
        public:
            Depth_Operations(const std::string& paramFilePath, const int width, const int height, const int cellSize);

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW


            /*
             * Create an point cloud organized by cells of cellSize*cellSize pixels
             *
             * in/out depthImage input depth image representation, transformed to align to rgb image at output
             * out organizedCloudArray 
             */
            void get_organized_cloud_array(cv::Mat& depthImage, Eigen::MatrixXf& organizedCloudArray);
            bool is_ok() const {return _isOk;};

        public: //getters
            float get_rgb_fx() const { return _fxRgb; }
            float get_rgb_fy() const { return _fyRgb; }
            float get_rgb_cx() const { return _cxRgb; }
            float get_rgb_cy() const { return _cyRgb; }

        protected:
            bool load_parameters(const std::string& parameterFilePath);
            void init_matrices();

        private:
            int _width;
            int _height;
            int _cellSize;
            bool _isOk;

            Eigen::MatrixXf _cloudArray;

            //cam parameters
            float _fxIr;
            float _fyIr;
            float _cxIr;
            float _cyIr;

            float _fxRgb;
            float _fyRgb;
            float _cxRgb;
            float _cyRgb;

            //camera parameters
            cv::Mat _Krgb;
            cv::Mat _Kir;
            cv::Mat _Rstereo;
            cv::Mat _Tstereo;

            //pre computation matrix
            cv::Mat_<float> _X;
            cv::Mat_<float> _Y;
            cv::Mat_<float> _Xt;
            cv::Mat_<float> _Yt;
            cv::Mat_<float> _Xpre;
            cv::Mat_<float> _Ypre;
            cv::Mat_<float> _U;   
            cv::Mat_<float> _V;
            cv::Mat_<int> _cellMap;
    };
}

#endif
