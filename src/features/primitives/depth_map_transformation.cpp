#include "depth_map_transformation.hpp"

#include "../../parameters.hpp"
#include "../../utils/angle_utils.hpp"
#include "../../utils/random.hpp"
#include "types.hpp"

#include <opencv2/core/eigen.hpp>
#include <tbb/parallel_for.h>

namespace rgbd_slam {
namespace features {
namespace primitives {

        Depth_Map_Transformation::Depth_Map_Transformation(const uint width, const uint height, const uint cellSize) 
            : 
                _width(width), _height(height), _cellSize(cellSize),
                _Xpre(height, width), _Ypre(height, width),
                _cellMap(height, width)
        {
            _isOk = false;
            _isOk = load_parameters();
            if(this->is_ok())
                init_matrices();
        }

        void Depth_Map_Transformation::get_organized_cloud_array(cv::Mat& depthImage, matrixf& organizedCloudArray) {
            if(not this->is_ok())
                return;

            // Backproject to point cloud (undistord depth image)
            cv::Mat_<float> X = _Xpre.mul(depthImage); 
            cv::Mat_<float> Y = _Ypre.mul(depthImage);

            // Transform depth image to rgb image space
            cv::Mat_<float> Xt = 
                static_cast<float>(_Rstereo.at<double>(0,0)) * X + 
                static_cast<float>(_Rstereo.at<double>(0,1)) * Y +
                static_cast<float>(_Rstereo.at<double>(0,2)) * depthImage + 
                static_cast<float>(_Tstereo.at<double>(0));
            cv::Mat_<float> Yt = 
                static_cast<float>(_Rstereo.at<double>(1,0)) * X +
                static_cast<float>(_Rstereo.at<double>(1,1)) * Y +
                static_cast<float>(_Rstereo.at<double>(1,2)) * depthImage +
                static_cast<float>(_Tstereo.at<double>(1));
            cv::Mat_<float> Zt = 
                static_cast<float>(_Rstereo.at<double>(2,0)) * X +
                static_cast<float>(_Rstereo.at<double>(2,1)) * Y + 
                static_cast<float>(_Rstereo.at<double>(2,2)) * depthImage + 
                static_cast<float>(_Tstereo.at<double>(2));

            double zMin = _Tstereo.at<double>(2);

            // will contain the projected depth image to rgb space
            depthImage = cv::Mat::zeros(_height, _width, CV_32F);
            
            #ifndef MAKE_DETERMINISTIC
            // parallel loop to speed up the process
            // USING THIS PARALLEL LOOP BREAKS THE RANDOM SEEDING
            tbb::parallel_for(uint(0), _height, [&](uint r){
            #else
            for(uint row = 0; row < _height; ++row) {
            #endif
                    // get depth map in camera space
                    const float* sx = Xt.ptr<float>(row);
                    const float* sy = Yt.ptr<float>(row);
                    const float* sz = Zt.ptr<float>(row);

                    for(uint column = 0; column < _width; column++){
                        const float z = sz[column];
                        if(z > zMin)
                        {
                            const float x = sx[column];
                            const float y = sy[column];
                            
                            const uint projCoordColumn = floor( (x/z) * _fxRgb + _cxRgb );
                            const uint projCoordRow = floor( (y/z) * _fyRgb + _cyRgb );
                            // keep projected coordinates that are in rgb image boundaries
                            if (projCoordColumn > 0 and projCoordRow > 0 and projCoordColumn < _width and projCoordRow < _height){
                                //set transformed depth image
                                depthImage.at<float>(projCoordRow, projCoordColumn) = z;

                                // set convertion matrix
                                const int id = _cellMap.at<int>(projCoordRow, projCoordColumn);
                                organizedCloudArray(id, 0) = x;
                                organizedCloudArray(id, 1) = y;
                                organizedCloudArray(id, 2) = z;
                            }
                        }
                    }
                }
            #ifndef MAKE_DETERMINISTIC
            );
            #endif
        }

        bool Depth_Map_Transformation::load_parameters() {
            // TODO check parameters 
            _fxIr = Parameters::get_camera_2_focal_x();
            _fyIr = Parameters::get_camera_2_focal_y();
            _cxIr = Parameters::get_camera_2_center_x();
            _cyIr = Parameters::get_camera_2_center_y();

            _fxRgb = Parameters::get_camera_1_focal_x();
            _fyRgb = Parameters::get_camera_1_focal_y();
            _cxRgb = Parameters::get_camera_1_center_x();
            _cyRgb = Parameters::get_camera_1_center_y();

            _Tstereo = cv::Mat(3, 1, CV_64F);
            _Tstereo.at<double>(0) = Parameters::get_camera_2_translation_x();
            _Tstereo.at<double>(1) = Parameters::get_camera_2_translation_y();
            _Tstereo.at<double>(2) = Parameters::get_camera_2_translation_z();

            const EulerAngles rotationEuler(
                    Parameters::get_camera_2_rotation_x(),
                    Parameters::get_camera_2_rotation_y(),
                    Parameters::get_camera_2_rotation_z()
                    );
            const matrix33 cameraRotation = utils::get_rotation_matrix_from_euler_angles(rotationEuler);
            cv::eigen2cv(cameraRotation, _Rstereo);
            return true;
        }

        /*
         *  Called after loading parameters to init matrices
         */
        void Depth_Map_Transformation::init_matrices() {
            const uint horizontalCellsCount = static_cast<uint>(_width / _cellSize);

            // Pre-computations for backprojection
            for (uint row = 0; row < _height; ++row){
                const uint cellR = floor(row / _cellSize);
                const uint localR = floor(row % _cellSize);

                for (uint colum = 0; colum < _width; ++colum){
                    // Not efficient but at this stage doesn t matter
                    _Xpre.at<float>(row, colum) = static_cast<float>((colum - _cxIr) / _fxIr);
                    _Ypre.at<float>(row, colum) = static_cast<float>((row - _cyIr) / _fyIr);

                    // Pre-computations for maping an image point cloud to a cache-friendly array where cell's local point clouds are contiguous
                    const uint cellC =  floor(colum / _cellSize);
                    const uint localC = floor(colum % _cellSize);
                    _cellMap.at<int>(row, colum) = (cellR * horizontalCellsCount + cellC) * pow(_cellSize, 2) + localR * _cellSize + localC;
                }
            }
        }


}
}
}
