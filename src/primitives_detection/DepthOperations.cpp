#include "DepthOperations.hpp"

using namespace primitiveDetection;

Depth_Operations::Depth_Operations(const std::string& parameterFilePath, const int width, const int height, const int cellSize) 
    : 
        width(width), height(height), cellSize(cellSize),
        cloudArray(width * height, 3),
        X(height, width), Y(height, width), Xt(height, width), Yt(height, width),
        Xpre(height, width), Ypre(height, width), 
        U(height, width), V(height, width), 
        cellMap(height, width)
{
    this->isOk = false;
    this->isOk = load_parameters(parameterFilePath);
    if(isOk)
        init_matrices();
}

void Depth_Operations::get_organized_cloud_array(cv::Mat& depthImage, Eigen::MatrixXf& organizedCloudArray) {
    if(not isOk)
        return;

    // Backproject to point cloud
    this->X = this->Xpre.mul(depthImage); 
    this->Y = this->Ypre.mul(depthImage);

    // The following transformation+projection is only necessary to visualize RGB with overlapped segments
    // Transform point cloud to color reference frame
    this->Xt = 
        ((float)this->Rstereo.at<double>(0,0)) * X + 
        ((float)this->Rstereo.at<double>(0,1)) * Y +
        ((float)this->Rstereo.at<double>(0,2)) * depthImage + 
        (float)this->Tstereo.at<double>(0);
    this->Yt = 
        ((float)this->Rstereo.at<double>(1,0)) * X +
        ((float)this->Rstereo.at<double>(1,1)) * Y +
        ((float)this->Rstereo.at<double>(1,2)) * depthImage +
        (float)this->Tstereo.at<double>(1);
    depthImage = 
        ((float)this->Rstereo.at<double>(2,0)) * X +
        ((float)this->Rstereo.at<double>(2,1)) * Y + 
        ((float)this->Rstereo.at<double>(2,2)) * depthImage + 
        (float)this->Tstereo.at<double>(2);

    double zMin = this->Tstereo.at<double>(2);

    // Project to image coordinates
    cv::divide(this->Xt, depthImage, this->U, 1);
    cv::divide(this->Yt, depthImage, this->V, 1);
    this->U = this->U * this->fxRgb + this->cxRgb;
    this->V = this->V * this->fyRgb + this->cyRgb;
    // Reusing U as cloud index
    //U = V*width + U + 0.5;

    cv::Mat outputDepth(height, width, CV_32F, 0.0);
    cloudArray.setZero();
    for(int r = 0; r < this->height; r++){
        float* sx = this->Xt.ptr<float>(r);
        float* sy = this->Yt.ptr<float>(r);
        float* sz = depthImage.ptr<float>(r);
        float* u_ptr = this->U.ptr<float>(r);
        float* v_ptr = this->V.ptr<float>(r);

        for(int c = 0; c < this->width; c++){
            float z = sz[c];
            float u = u_ptr[c];
            float v = v_ptr[c];
            if(z > zMin and u > 0 and v > 0 and u < this->width and v < this->height){
                //set transformed depth image
                outputDepth.at<float>(v, c) = depthImage.at<float>(v, u);
                int id = floor(v) * this->width + u;
                this->cloudArray(id, 0) = sx[c];
                this->cloudArray(id, 1) = sy[c];
                this->cloudArray(id, 2) = z;

            }
        }
    }

    //project cloud point by cells
    int mxn = this->width * this->height;
    int mxn2 = 2 * mxn;
    for(int r = 0, it = 0; r < this->height; r++){
        int* cellMapPtr = this->cellMap.ptr<int>(r);
        for(int c = 0; c < width; c++, it++){
            int id = cellMapPtr[c];
            organizedCloudArray(id) = this->cloudArray(it);
            organizedCloudArray(mxn + id) = this->cloudArray(mxn + it);
            organizedCloudArray(mxn2 + id) = this->cloudArray(mxn2 + it);
        }
    }
    depthImage = outputDepth;
}

bool Depth_Operations::load_parameters(const std::string& parameterFilePath) {
    cv::FileStorage fs(parameterFilePath, cv::FileStorage::READ);
    if (fs.isOpened()) {
        fs["RGB_intrinsic_params"] >> Krgb;
        //fs["RGB_distortion_coefficients"] >> distCoefficientsRGB;
        fs["IR_intrinsic_params"] >> Kir;
        //fs["IR_distortion_coefficients"] >> distCoefficientsIR;
        fs["Rotation"] >> Rstereo;
        fs["Translation"] >> Tstereo;
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
    int horizontalCellsCount = this->width / this->cellSize;

    this->fxIr = this->Kir.at<double>(0,0);
    this->fyIr = this->Kir.at<double>(1,1);
    this->cxIr = this->Kir.at<double>(0,2);
    this->cyIr = this->Kir.at<double>(1,2);
    this->fxRgb = this->Krgb.at<double>(0,0);
    this->fyRgb = this->Krgb.at<double>(1,1);
    this->cxRgb = this->Krgb.at<double>(0,2);
    this->cyRgb = this->Krgb.at<double>(1,2);

    // Pre-computations for backprojection
    for (int r = 0; r < this->height; r++){
        for (int c = 0; c < this->width; c++){
            // Not efficient but at this stage doesn t matter
            this->Xpre.at<float>(r, c) = (c - this->cxIr) / this->fxIr; 
            this->Ypre.at<float>(r, c) = (r - this->cyIr) / this->fyIr;
        }
    }

    // Pre-computations for maping an image point cloud to a cache-friendly array where cell's local point clouds are contiguous
    for (int r = 0; r < this->height; r++){
        int cellR = r / this->cellSize;
        int localR = r % this->cellSize;

        for (int c = 0; c < this->width; c++){
            int cellC = c / this->cellSize;
            int localC = c % this->cellSize;
            this->cellMap.at<int>(r, c) = (cellR * horizontalCellsCount + cellC) * this->cellSize * this->cellSize + localR * this->cellSize + localC;
        }
    }
}





