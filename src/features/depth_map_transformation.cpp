#include "depth_map_transformation.hpp"
#include "parameters.hpp"
#include "utils.hpp"

#include <opencv2/core/eigen.hpp>
#include <tbb/parallel_for.h>

namespace rgbd_slam {
namespace features {
namespace primitives {

        Depth_Map_Transformation::Depth_Map_Transformation(const uint width, const uint height, const uint cellSize) 
            : 
                _width(width), _height(height), _cellSize(cellSize),
                _cloudArray(width * height, 3),
                _X(height, width), _Y(height, width), _Xt(height, width), _Yt(height, width),
                _Xpre(height, width), _Ypre(height, width), 
                _U(height, width), _V(height, width), 
                _cellMap(height, width)
        {
            _isOk = false;
            _isOk = load_parameters();
            if(this->is_ok())
                init_matrices();
        }

        void Depth_Map_Transformation::get_organized_cloud_array(cv::Mat& depthImage, Eigen::MatrixXf& organizedCloudArray) {
            if(not this->is_ok())
                return;

            // Backproject to point cloud
            _X = _Xpre.mul(depthImage); 
            _Y = _Ypre.mul(depthImage);

            // The following transformation+projection is only necessary to visualize RGB with overlapped segments
            // Transform point cloud to color reference frame
            _Xt = 
                static_cast<float>(_Rstereo.at<double>(0,0)) * _X + 
                static_cast<float>(_Rstereo.at<double>(0,1)) * _Y +
                static_cast<float>(_Rstereo.at<double>(0,2)) * depthImage + 
                static_cast<float>(_Tstereo.at<double>(0));
            _Yt = 
                static_cast<float>(_Rstereo.at<double>(1,0)) * _X +
                static_cast<float>(_Rstereo.at<double>(1,1)) * _Y +
                static_cast<float>(_Rstereo.at<double>(1,2)) * depthImage +
                static_cast<float>(_Tstereo.at<double>(1));
            depthImage = 
                static_cast<float>(_Rstereo.at<double>(2,0)) * _X +
                static_cast<float>(_Rstereo.at<double>(2,1)) * _Y + 
                static_cast<float>(_Rstereo.at<double>(2,2)) * depthImage + 
                static_cast<float>(_Tstereo.at<double>(2));

            double zMin = _Tstereo.at<double>(2);

            // Project to image coordinates
            cv::divide(_Xt, depthImage, _U, 1);
            cv::divide(_Yt, depthImage, _V, 1);
            _U = _U * _fxRgb + _cxRgb;
            _V = _V * _fyRgb + _cyRgb;
            // Reusing U as cloud index
            //U = V*width + U + 0.5;

            cv::Mat outputDepth = cv::Mat::zeros(_height, _width, CV_32F);
            _cloudArray.setZero();
            // parallel loop to speed up the process
            tbb::parallel_for(uint(0), _height, [&](uint r){
                    float* sx = _Xt.ptr<float>(r);
                    float* sy = _Yt.ptr<float>(r);
                    float* sz = depthImage.ptr<float>(r);
                    float* u_ptr = _U.ptr<float>(r);
                    float* v_ptr = _V.ptr<float>(r);

                    for(uint c = 0; c < _width; c++){
                        float z = sz[c];
                        float u = u_ptr[c];
                        float v = v_ptr[c];
                        if(z > zMin and u > 0 and v > 0 and u < _width and v < _height){
                            //set transformed depth image
                            outputDepth.at<float>(v, c) = z; 
                            int id = floor(v) * _width + u;
                            _cloudArray(id, 0) = sx[c];
                            _cloudArray(id, 1) = sy[c];
                            _cloudArray(id, 2) = z;
                        }
                    }
                }
            );

            //project cloud point by cells
            uint mxn = _width * _height;
            uint mxn2 = 2 * mxn;
            for(uint r = 0, it = 0; r < _height; r++){
                int* cellMapPtr = _cellMap.ptr<int>(r);
                for(uint c = 0; c < _width; c++, it++){
                    int id = cellMapPtr[c];
                    organizedCloudArray(id) = _cloudArray(it);
                    organizedCloudArray(mxn + id) = _cloudArray(mxn + it);
                    organizedCloudArray(mxn2 + id) = _cloudArray(mxn2 + it);
                }
            }
            depthImage = outputDepth;
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
            uint horizontalCellsCount = static_cast<uint>(_width / _cellSize);

            // Pre-computations for backprojection
            for (uint r = 0; r < _height; r++){
                for (uint c = 0; c < _width; c++){
                    // Not efficient but at this stage doesn t matter
                    _Xpre.at<float>(r, c) = static_cast<float>((c - _cxIr) / _fxIr); 
                    _Ypre.at<float>(r, c) = static_cast<float>((r - _cyIr) / _fyIr);
                }
            }

            // Pre-computations for maping an image point cloud to a cache-friendly array where cell's local point clouds are contiguous
            for (uint r = 0; r < _height; r++){
                uint cellR = static_cast<uint>(r / _cellSize);
                uint localR = static_cast<uint>(r % _cellSize);

                for (uint c = 0; c < _width; c++){
                    uint cellC =  static_cast<uint>(c / _cellSize);
                    uint localC = static_cast<uint>(c % _cellSize);
                    _cellMap.at<int>(r, c) = (cellR * horizontalCellsCount + cellC) * _cellSize * _cellSize + localR * _cellSize + localC;
                }
            }
        }


}
}
}
