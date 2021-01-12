#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "DepthOperations.hpp"
#include "PlaneDetection.hpp"
#include "PlaneSegment.hpp"

using namespace planeDetection;
using namespace std;

const float COS_ANGLE_MAX = cos(M_PI/12.0);
const float MAX_MERGE_DIST = 100;//50.0f;
const unsigned int PATCH_SIZE = 20;   //depth grid cell size


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


    std::stringstream depthImagePath, rgbImgPath;
    depthImagePath << dataPath.str() << "depth_0.png";

    int width, height;
    cv::Mat dImage = cv::imread(depthImagePath.str(), cv::IMREAD_ANYDEPTH);
    if(dImage.data) {
        width = dImage.cols;
        height = dImage.rows;
    }
    else {
        std::cout << "Error loading first depth image at " << depthImagePath.str() << std::endl;
        return -1;
    }

    // Get intrinsics parameters
    std::stringstream calibPath;
    calibPath << dataPath.str() << "calib_params.xml";

    Depth_Operations depthOps(calibPath.str(), width, height, PATCH_SIZE);
    if (not depthOps.is_ok()) {
        exit(-1);
    }

    Plane_Detection detector(height, width, PATCH_SIZE, COS_ANGLE_MAX, MAX_MERGE_DIST, useCylinderFitting);

    // Populate with random color codes
    std::vector<cv::Vec3b> color_code = get_color_vector();

    //organized 3D depth image
    Eigen::MatrixXf cloudArrayOrganized(width * height,3);

    int i = 1;
    double meanTreatmentTime = 0.0;
    double maxTreatTime = 0.0;
    std::cout << "Starting extraction" << std::endl;
    while(1) {
        depthImagePath.str("");
        depthImagePath << dataPath.str() << "depth_" << i << ".png";

        rgbImgPath.str("");
        rgbImgPath << dataPath.str() << "rgb_"<< i <<".png";

        cv::Mat depthImage = cv::imread(depthImagePath.str(), cv::IMREAD_ANYDEPTH);
        cv::Mat rgbImage = cv::imread(rgbImgPath.str(), cv::IMREAD_COLOR);
        if (!depthImage.data or !rgbImage.data)
            break;
        std::cout << "Read frame " << i << std::endl;

        depthImage.convertTo(depthImage, CV_32F);


        //project depth image in an organized cloud
        depthOps.get_organized_cloud_array(depthImage, cloudArrayOrganized);


        double t1 = cv::getTickCount();

        vector<Plane_Segment> planeParams;
        vector<Cylinder_Segment> cylinderParams;
        cv::Mat_<uchar> seg_output = cv::Mat_<uchar>(height, width, uchar(0));

        // Run plane and cylinder detection 
        detector.find_plane_regions(cloudArrayOrganized, planeParams, cylinderParams, seg_output);

        //get elapsed time
        double time_elapsed = (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
        meanTreatmentTime += time_elapsed;
        maxTreatTime = max(maxTreatTime, time_elapsed);

        //convert to 8 bits
        double min, max;
        cv::minMaxLoc(depthImage, &min, &max);
        if (min != max){ 
            depthImage -= min;
            depthImage.convertTo(depthImage, CV_8U, 255.0/(max-min));
        }
        cv::Mat segRgb(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat segDepth(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
        for(int r = 0; r < height; r++){
            uchar* dColor = segRgb.ptr<uchar>(r);
            uchar* dDepth = segDepth.ptr<uchar>(r);
            uchar* sCode = seg_output.ptr<uchar>(r);
            uchar* srgb = rgbImage.ptr<uchar>(r);
            uchar* sdepth = depthImage.ptr<uchar>(r);
            for(int c = 0; c < width; c++){
                int code = seg_output(r, c);
                if (code > 0){
                    dColor[c*3] =   color_code[code-1][0]/2 + srgb[0]/2;
                    dColor[c*3+1] = color_code[code-1][1]/2 + srgb[1]/2;
                    dColor[c*3+2] = color_code[code-1][2]/2 + srgb[2]/2;
                }else {
                    dColor[c*3]  =  srgb[0]/2;
                    dColor[c*3+1] = srgb[1]/2;
                    dColor[c*3+2] = srgb[2]/2;
                }

                if(sdepth[0] == 0) {
                    dDepth[c*3]  =  sdepth[0];
                    dDepth[c*3+1] = sdepth[1];
                    dDepth[c*3+2] = sdepth[2];
                }
                else {
                    dDepth[c*3]  =  255 - sdepth[0];
                    dDepth[c*3+1] = 255 - sdepth[1];
                    dDepth[c*3+2] = 255 - sdepth[2];
                }
                sCode++; srgb += 3; sdepth += 1; 
            }
        }
        cv::hconcat(segDepth, segRgb, segRgb);

        // Show frame rate and labels
        cv::rectangle(segRgb,  cv::Point(0,0),cv::Point(width * 2, 20), cv::Scalar(0,0,0),-1);
        std::stringstream fps;
        fps << (int)(1/time_elapsed+0.5) << " fps";
        cv::putText(segRgb, fps.str(), cv::Point(15,15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255,1));

        // show cylinder labels
        if (cylinderParams.size() > 0){
            std::stringstream text;
            text << "Cylinders:";
            cv::putText(segRgb, text.str(), cv::Point(width, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 1));
            for(unsigned int j = 0; j < cylinderParams.size(); j += 1){
                cv::rectangle(segRgb,  cv::Point(width + 80 + 15 * j, 6),
                        cv::Point(width + 90 + 15 * j, 16), 
                        cv::Scalar(
                            color_code[CYLINDER_CODE_OFFSET + j][0],
                            color_code[CYLINDER_CODE_OFFSET + j][1],
                            color_code[CYLINDER_CODE_OFFSET + j][2]),
                        -1);
            }
        }

        cv::imshow("Seg", segRgb);
        char key = cv::waitKey(5);
        if (key == 'q')
            break;
        if (key == 'p' or cylinderParams.size() > 2)
            cv::waitKey(-1); //wait until any key is pressed
        i++;
    }
    std::cout << "Mean plane treatment time is " << meanTreatmentTime/i << std::endl;
    std::cout << "max treat time is " << maxTreatTime << std::endl;
    //    cv::destroyAllWindows();
    exit(0);
}
