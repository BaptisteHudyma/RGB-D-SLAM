#include "DepthOperations.hpp"

namespace primitiveDetection {

    Depth_Operations::Depth_Operations(const std::string& parameterFilePath, const int width, const int height, const int cellSize) 
        : 
            _width(width), _height(height), _cellSize(cellSize),
            _cloudArray(width * height, 3),
            _X(height, width), _Y(height, width), _Xt(height, width), _Yt(height, width),
            _Xpre(height, width), _Ypre(height, width), 
            _U(height, width), _V(height, width), 
            _cellMap(height, width)
    {
        _isOk = false;
        _isOk = load_parameters(parameterFilePath);
        if(this->is_ok())
            init_matrices();
    }

    void Depth_Operations::get_organized_cloud_array(cv::Mat& depthImage, Eigen::MatrixXf& organizedCloudArray) {
        if(not this->is_ok())
            return;

        // Backproject to point cloud
        _X = _Xpre.mul(depthImage); 
        _Y = _Ypre.mul(depthImage);

        // The following transformation+projection is only necessary to visualize RGB with overlapped segments
        // Transform point cloud to color reference frame
        _Xt = 
            ((float)_Rstereo.at<double>(0,0)) * _X + 
            ((float)_Rstereo.at<double>(0,1)) * _Y +
            ((float)_Rstereo.at<double>(0,2)) * depthImage + 
            (float)_Tstereo.at<double>(0);
        _Yt = 
            ((float)_Rstereo.at<double>(1,0)) * _X +
            ((float)_Rstereo.at<double>(1,1)) * _Y +
            ((float)_Rstereo.at<double>(1,2)) * depthImage +
            (float)_Tstereo.at<double>(1);
        depthImage = 
            ((float)_Rstereo.at<double>(2,0)) * _X +
            ((float)_Rstereo.at<double>(2,1)) * _Y + 
            ((float)_Rstereo.at<double>(2,2)) * depthImage + 
            (float)_Tstereo.at<double>(2);

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
        for(int r = 0; r < _height; r++){
            float* sx = _Xt.ptr<float>(r);
            float* sy = _Yt.ptr<float>(r);
            float* sz = depthImage.ptr<float>(r);
            float* u_ptr = _U.ptr<float>(r);
            float* v_ptr = _V.ptr<float>(r);

            for(int c = 0; c < _width; c++){
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

        //project cloud point by cells
        int mxn = _width * _height;
        int mxn2 = 2 * mxn;
        for(int r = 0, it = 0; r < _height; r++){
            int* cellMapPtr = _cellMap.ptr<int>(r);
            for(int c = 0; c < _width; c++, it++){
                int id = cellMapPtr[c];
                organizedCloudArray(id) = _cloudArray(it);
                organizedCloudArray(mxn + id) = _cloudArray(mxn + it);
                organizedCloudArray(mxn2 + id) = _cloudArray(mxn2 + it);
            }
        }
        depthImage = outputDepth;
    }

    bool Depth_Operations::load_parameters(const std::string& parameterFilePath) {
        cv::FileStorage fs(parameterFilePath, cv::FileStorage::READ);
        if (fs.isOpened()) {
            fs["RGB_intrinsic_params"] >> _Krgb;
            //fs["RGB_distortion_coefficients"] >> distCoefficientsRGB;
            fs["IR_intrinsic_params"] >> _Kir;
            //fs["IR_distortion_coefficients"] >> distCoefficientsIR;
            fs["Rotation"] >> _Rstereo;
            fs["Translation"] >> _Tstereo;
            fs.release();
            return true;
        }else{
            std::cerr << "Calibration file " << parameterFilePath<< " missing" << std::endl;
            return false;
        }
        fs.release();
    }

    /*
     *  Called after loading parameters to init matrices
     */
    void Depth_Operations::init_matrices() {
        int horizontalCellsCount = _width / _cellSize;

        _fxIr = _Kir.at<double>(0,0);
        _fyIr = _Kir.at<double>(1,1);
        _cxIr = _Kir.at<double>(0,2);
        _cyIr = _Kir.at<double>(1,2);
        _fxRgb = _Krgb.at<double>(0,0);
        _fyRgb = _Krgb.at<double>(1,1);
        _cxRgb = _Krgb.at<double>(0,2);
        _cyRgb = _Krgb.at<double>(1,2);

        // Pre-computations for backprojection
        for (int r = 0; r < _height; r++){
            for (int c = 0; c < _width; c++){
                // Not efficient but at this stage doesn t matter
                _Xpre.at<float>(r, c) = (c - _cxIr) / _fxIr; 
                _Ypre.at<float>(r, c) = (r - _cyIr) / _fyIr;
            }
        }

        // Pre-computations for maping an image point cloud to a cache-friendly array where cell's local point clouds are contiguous
        for (int r = 0; r < _height; r++){
            int cellR = r / _cellSize;
            int localR = r % _cellSize;

            for (int c = 0; c < _width; c++){
                int cellC = c / _cellSize;
                int localC = c % _cellSize;
                _cellMap.at<int>(r, c) = (cellR * horizontalCellsCount + cellC) * _cellSize * _cellSize + localR * _cellSize + localC;
            }
        }
    }



}
