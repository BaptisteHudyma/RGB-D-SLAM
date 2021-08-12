#ifndef DEPTH_OPERATIONS_H
#define DEPTH_OPERATIONS_H 

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace rgbd_slam {
namespace primitiveDetection {

    /**
      * \brief Handles operations on the initial depth image, to transform it on a connected cloud points. It also handles the loading of the camera parameters from the configuration file
      */
    class Depth_Operations {
        public:
            /**
              * \param[in] paramFilePath Path of the file containing the camera parameters
              * \param[in] width Depth image width (constant)
              * \param[in] height Depth image height (constant)
              * \param[in] cellSize Size of the cloud point division (> 0)
              */
            Depth_Operations(const std::string& paramFilePath, unsigned int width, unsigned int height, unsigned int cellSize);

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW


            /**
             * \brief Create an point cloud organized by cells of cellSize*cellSize pixels
             *
             * \param[in, out] depthImage Input depth image representation, transformed to align to rgb image at output
             * \param[out] organizedCloudArray A cloud point divided in blocs of cellSize * cellSize
             */
            void get_organized_cloud_array(cv::Mat& depthImage, Eigen::MatrixXf& organizedCloudArray);

            /**
              * \brief Controls the state of this class.
              *
              * \return False if the camera parameters could not be loaded
              */
            bool is_ok() const {return _isOk;};

        public: //getters
            float get_rgb_fx() const { return _fxRgb; }
            float get_rgb_fy() const { return _fyRgb; }
            float get_rgb_cx() const { return _cxRgb; }
            float get_rgb_cy() const { return _cyRgb; }

        protected:
            /**
              * \brief Loads the camera intrinsic parameters
              */
            bool load_parameters(const std::string& parameterFilePath);

            /**
              * \brief Must be called after load_parameters. Fills the computation matrices
              */
            void init_matrices();

        private:
            unsigned int _width;
            unsigned int _height;
            unsigned int _cellSize;
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
}

#endif
