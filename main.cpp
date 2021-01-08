#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "PlaneDetection.hpp"

#define BLOC_SIZE 20    //20*20 divided depth bloc size
#define PATCH_SIZE 20

int main() {
    std::string dataPath = "./data/";

    std::stringstream depthImagePath;
    depthImagePath << dataPath << "depth_0.png";

    int width, height;
    cv::Mat depthImage;

    depthImage = cv::imread(depthImagePath.str(), cv::IMREAD_ANYDEPTH);
    if(depthImage.data) {
        width = depthImage.cols;
        height = depthImage.rows;
    }
    else {
        std::cout << "Error loading first depth image at " << depthImagePath.str() << std::endl;
        return -1;
    }
    
    planeDetection::Plane_Detection pd(width, height, BLOC_SIZE);

    int nrHorizontalCells = width/PATCH_SIZE;
    int nrVerticalCells = height/PATCH_SIZE;

    int i = 1;
    while(1) {
        depthImagePath.str("");
        depthImagePath << dataPath << "depth_" << i << ".png";

        depthImage = cv::imread(depthImagePath.str(), cv::IMREAD_ANYDEPTH);
        if (!depthImage.data)
            break;

        depthImage.convertTo(depthImage, CV_32F);
        depthImage /= 1024;

        cv::imshow("Seg", depthImage);
        cv::waitKey(1);
        i++;
    }

    return 0;
}
