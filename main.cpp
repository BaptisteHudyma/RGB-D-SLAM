#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "PlaneDetection.hpp"
#include "PlaneSegment.hpp"

#define BLOC_SIZE 20    //20*20 divided depth bloc size
#define PATCH_SIZE 20

using namespace planeDetection;
using namespace std;

const float COS_ANGLE_MAX = cos(M_PI/12);
const float MAX_MERGE_DIST = 50.0f;

bool loadCalibParameters(std::string filepath, cv::Mat& intrinsics_rgb, cv::Mat& dist_coeffs_rgb, cv::Mat& intrinsics_ir, cv::Mat& dist_coeffs_ir, cv::Mat& R, cv::Mat& T){
    cv::FileStorage fs(filepath, cv::FileStorage::READ);
    if (fs.isOpened()){
        fs["RGB_intrinsic_params"] >> intrinsics_rgb;
        fs["RGB_distortion_coefficients"] >> dist_coeffs_rgb;
        fs["IR_intrinsic_params"] >> intrinsics_ir;
        fs["IR_distortion_coefficients"] >> dist_coeffs_ir;
        fs["Rotation"] >> R;
        fs["Translation"] >> T;
        fs.release();
        return true;
    }else{
        std::cerr << "Calibration file " << filepath << " missing" << std::endl;
        return false;
    }
    fs.release();
}

void projectPointCloud(cv::Mat& X, cv::Mat& Y, cv::Mat& Z, cv::Mat& U, cv::Mat& V, float fx_rgb, float fy_rgb, float cx_rgb, float cy_rgb, double z_min, Eigen::MatrixXf& cloud_array){

    int width = X.cols;
    int height = X.rows;

    // Project to image coordinates
    cv::divide(X, Z, U, 1);
    cv::divide(Y, Z, V, 1);
    U = U * fx_rgb + cx_rgb;
    V = V * fy_rgb + cy_rgb;
    // Reusing U as cloud index
    //U = V*width + U + 0.5;

    for(int r=0; r< height; r++){
        float* sx = X.ptr<float>(r);
        float* sy = Y.ptr<float>(r);
        float* sz = Z.ptr<float>(r);
        float* u_ptr = U.ptr<float>(r);
        float* v_ptr = V.ptr<float>(r);
        for(int c=0; c< width; c++){
            float z = sz[c];
            float u = u_ptr[c];
            float v = v_ptr[c];
            if(z > z_min and u > 0 and v > 0 and u < width and v < height){
                int id = floor(v) * width + u;
                cloud_array(id, 0) = sx[c];
                cloud_array(id, 1) = sy[c];
                cloud_array(id, 2) = z;
            }
        }
    }
}

void organizePointCloudByCell(Eigen::MatrixXf & cloud_in, Eigen::MatrixXf & cloud_out, cv::Mat & cell_map){

    int width = cell_map.cols;
    int height = cell_map.rows;
    int mxn = width * height;
    int mxn2 = 2 * mxn;

    int it(0);
    for(int r = 0; r < height; r++){
        int* cell_map_ptr = cell_map.ptr<int>(r);
        for(int c = 0; c < width; c++){
            int id = cell_map_ptr[c];
            *(cloud_out.data() + id) = *(cloud_in.data() + it);
            *(cloud_out.data() + mxn + id) = *(cloud_in.data() + mxn + it);
            *(cloud_out.data() + mxn2 + id) = *(cloud_in.data() + mxn2 + it);
            it++;
        }
    }
}

std::vector<cv::Vec3b> get_color_vector() {
    std::vector<cv::Vec3b> color_code;
    for(int i = 0; i < 100; i++){
        cv::Vec3b color;
        color[0] = rand() % 255;
        color[1] = rand() % 255;
        color[2] = rand() % 255;
        color_code.push_back(color);
    }

    // Add specific colors for planes
    color_code[0][0] = 0; color_code[0][1] = 0; color_code[0][2] = 255;
    color_code[1][0] = 255; color_code[1][1] = 0; color_code[1][2] = 204;
    color_code[2][0] = 255; color_code[2][1] = 100; color_code[2][2] = 0;
    color_code[3][0] = 0; color_code[3][1] = 153; color_code[3][2] = 255;
    // Add specific colors for cylinders
    color_code[50][0] = 178; color_code[50][1] = 255; color_code[50][2] = 0;
    color_code[51][0] = 255; color_code[51][1] = 0; color_code[51][2] = 51;
    color_code[52][0] = 0; color_code[52][1] = 255; color_code[52][2] = 51;
    color_code[53][0] = 153; color_code[53][1] = 0; color_code[53][2] = 255;
    return color_code;
}




int main(int argc, char** argv) {
    //std::string dataPath = "./data/yoga_mat/";
    std::stringstream dataPath;
    dataPath << "./data/";
    bool useCylinderFitting = false;
    if(argc > 1) {
        if(strcmp(argv[1], "-h") == 0) {
            std::cout << "Use of the prog:" << std::endl;
            std::cout << "\tFirst argument is the name of the data folder to parse" << std::endl;
            std::cout << "\t1: use cylinder fitting, 0 or nothing: do not use it" << std::endl; 
        }
        dataPath << argv[1] << "/";

        if (argc > 2) {
            useCylinderFitting = atoi(argv[2]) > 0;
        }
    }
    else {
        dataPath << "tunnel/";
    }

    // Get intrinsics
    cv::Mat K_rgb, K_ir, dist_coeffs_rgb, dist_coeffs_ir, R_stereo, t_stereo;
    std::stringstream calib_path;
    calib_path << dataPath.str() << "calib_params.xml";
    bool res = loadCalibParameters(calib_path.str(), K_rgb, dist_coeffs_rgb, K_ir, dist_coeffs_ir, R_stereo, t_stereo);
    if(not res) {
        exit(-1);
    }

    float fx_ir = K_ir.at<double>(0,0);
    float fy_ir = K_ir.at<double>(1,1);
    float cx_ir = K_ir.at<double>(0,2);
    float cy_ir = K_ir.at<double>(1,2);
    float fx_rgb = K_rgb.at<double>(0,0);
    float fy_rgb = K_rgb.at<double>(1,1);
    float cx_rgb = K_rgb.at<double>(0,2);
    float cy_rgb = K_rgb.at<double>(1,2);


    std::stringstream depthImagePath, rgbImgPath;
    depthImagePath << dataPath.str() << "depth_0.png";

    int width, height;
    cv::Mat depthImage;
    cv::Mat rgbImage;

    depthImage = cv::imread(depthImagePath.str(), cv::IMREAD_ANYDEPTH);
    if(depthImage.data) {
        width = depthImage.cols;
        height = depthImage.rows;
    }
    else {
        std::cout << "Error loading first depth image at " << depthImagePath.str() << std::endl;
        return -1;
    }

    planeDetection::Plane_Detection detector(height, width, PATCH_SIZE, COS_ANGLE_MAX, MAX_MERGE_DIST, useCylinderFitting);

    std::cout << "Starting extraction" << std::endl;

    int horizontalCellsCount = width / PATCH_SIZE;


    // Populate with random color codes
    std::vector<cv::Vec3b> color_code = get_color_vector();


    // Pre-computations for backprojection
    cv::Mat_<float> X_pre(height, width);
    cv::Mat_<float> Y_pre(height, width);
    cv::Mat_<float> U(height, width);
    cv::Mat_<float> V(height, width);
    for (int r = 0; r < height; r++){
        for (int c = 0; c < width; c++){
            // Not efficient but at this stage doesn t matter
            X_pre.at<float>(r, c) = (c - cx_ir) / fx_ir; 
            Y_pre.at<float>(r, c) = (r - cy_ir) / fy_ir;
        }
    }

    // Pre-computations for maping an image point cloud to a cache-friendly array where cell's local point clouds are contiguous
    cv::Mat_<int> cell_map(height,width);
    for (int r = 0;r < height; r++){
        int cell_r = r / PATCH_SIZE;
        int local_r = r % PATCH_SIZE;
        for (int c = 0;c < width; c++){
            int cell_c = c / PATCH_SIZE;
            int local_c = c % PATCH_SIZE;
            cell_map.at<int>(r, c) = (cell_r * horizontalCellsCount + cell_c) * PATCH_SIZE * PATCH_SIZE + local_r * PATCH_SIZE + local_c;
        }
    }

    cv::Mat_<float> X(height, width);
    cv::Mat_<float> Y(height, width);
    cv::Mat_<float> X_t(height, width);
    cv::Mat_<float> Y_t(height, width);
    Eigen::MatrixXf cloud_array(width * height,3);
    Eigen::MatrixXf cloud_array_organized(width * height,3);

    int i = 1;
    double meanTreatmentTime = 0.0;
    double maxTreatTime = 0.0;
    std::cout << "Starting extraction" << std::endl;
    while(1) {
        depthImagePath.str("");
        depthImagePath << dataPath.str() << "depth_" << i << ".png";

        rgbImgPath.str("");
        rgbImgPath << dataPath.str() << "rgb_"<< i <<".png";

        depthImage = cv::imread(depthImagePath.str(), cv::IMREAD_ANYDEPTH);
        rgbImage = cv::imread(rgbImgPath.str(), cv::IMREAD_COLOR);
        if (!depthImage.data or !rgbImage.data)
            break;
        std::cout << "Read frame " << i << std::endl;

        depthImage.convertTo(depthImage, CV_32F);

        // Backproject to point cloud
        X = X_pre.mul(depthImage); 
        Y = Y_pre.mul(depthImage);
        cloud_array.setZero();

        // The following transformation+projection is only necessary to visualize RGB with overlapped segments
        // Transform point cloud to color reference frame
        X_t = ((float)R_stereo.at<double>(0,0)) * X + ((float)R_stereo.at<double>(0,1)) * Y + ((float)R_stereo.at<double>(0,2)) * depthImage + (float)t_stereo.at<double>(0);
        Y_t = ((float)R_stereo.at<double>(1,0)) * X + ((float)R_stereo.at<double>(1,1)) * Y + ((float)R_stereo.at<double>(1,2)) * depthImage + (float)t_stereo.at<double>(1);
        depthImage = ((float)R_stereo.at<double>(2,0)) * X + ((float)R_stereo.at<double>(2,1)) * Y + ((float)R_stereo.at<double>(2,2)) * depthImage + (float)t_stereo.at<double>(2);

        projectPointCloud(X_t, Y_t, depthImage, U, V, fx_rgb, fy_rgb, cx_rgb, cy_rgb, t_stereo.at<double>(2), cloud_array);

        cv::Mat seg_rz(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat_<uchar> seg_output = cv::Mat_<uchar>(height, width, uchar(0));


        // Run plane and cylinder detection 

        vector<Plane_Segment> planeParams;
        vector<Cylinder_Segment> cylinderParams;

        double t1 = cv::getTickCount();
        organizePointCloudByCell(cloud_array, cloud_array_organized, cell_map);

        detector.find_plane_regions(cloud_array_organized, planeParams, cylinderParams, seg_output);

        double t2 = cv::getTickCount();
        double time_elapsed = (t2-t1)/(double)cv::getTickFrequency();
        meanTreatmentTime += time_elapsed;
        maxTreatTime = max(maxTreatTime, time_elapsed);
        //convert to 8 bits

        double min, max;
        cv::minMaxLoc(depthImage, &min, &max);
        if (min != max){ 
            depthImage -= min;
            depthImage.convertTo(depthImage, CV_8U, 255.0/(max-min));
        }
        for(int r = 0; r < height; r++){
            uchar* dColor = seg_rz.ptr<uchar>(r);
            uchar* sCode = seg_output.ptr<uchar>(r);
            uchar* srgb = rgbImage.ptr<uchar>(r);
            //uchar* srgb = depthImage.ptr<uchar>(r);
            for(int c = 0; c < width; c++){
                int code = seg_output(r, c);
                if (code > 0){
                    dColor[c*3] =   color_code[code-1][0]/2 + srgb[0]/2;
                    dColor[c*3+1] = color_code[code-1][1]/2 + srgb[1]/2;
                    dColor[c*3+2] = color_code[code-1][2]/2 + srgb[2]/2;
                }else{
                    dColor[c*3]  =  srgb[0]/2;
                    dColor[c*3+1] = srgb[1]/2;
                    dColor[c*3+2] = srgb[2]/2;
                }
                sCode++; srgb+=3; 
            }
        }

        // Show frame rate and labels
        cv::rectangle(seg_rz,  cv::Point(0,0),cv::Point(width,20), cv::Scalar(0,0,0),-1);
        std::stringstream fps;
        fps << (int)(1/time_elapsed+0.5) << " fps";
        cv::putText(seg_rz, fps.str(), cv::Point(15,15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255,1));

        // show cylinder labels
        if (cylinderParams.size() > 0){
            std::stringstream text;
            text << "Cylinders:";
            cv::putText(seg_rz, text.str(), cv::Point(width/2,15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 1));
            for(unsigned int j = 0; j < cylinderParams.size(); j += 1){
                cv::rectangle(seg_rz,  cv::Point(width/2 + 80 + 15 * j, 6),
                        cv::Point(width / 2 + 90 + 15 * j, 16), 
                        cv::Scalar(
                            color_code[CYLINDER_CODE_OFFSET + j][0],
                            color_code[CYLINDER_CODE_OFFSET + j][1],
                            color_code[CYLINDER_CODE_OFFSET + j][2]),
                        -1);
            }
        }

        cv::imshow("Seg", seg_rz);
        cv::waitKey(1);
        i++;
    }
    std::cout << "Mean plane treatment time is " << meanTreatmentTime/i << std::endl;
    std::cout << "max treat time is " << maxTreatTime << std::endl;
    //    cv::destroyAllWindows();
    exit(0);
}
