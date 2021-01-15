#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor.hpp>
#include <Eigen/Dense>

#include "DepthOperations.hpp"
#include "PlaneDetection.hpp"
#include "PlaneSegment.hpp"
#include "Parameters.hpp"
#include "LineSegmentDetector.hpp"

using namespace planeDetection;
using namespace cv::line_descriptor;
using namespace std;


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


int main(int argc, char* argv[]) {

    const cv::String keys = 
        "{help h usage ?  |      | print this message     }"
        "{f folder   |<none>| folder to parse        }"
        "{c cylinder |  1   | Use cylinder detection }"
        "{i index    |  0   | First image to parse   }"
        ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Plane Detection v1");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    std::stringstream dataPath;
    dataPath << parser.get<cv::String>("f") << "/";
    bool useCylinderFitting = parser.get<bool>("c");
    int startIndex = parser.get<int>("i");

    if (!parser.check()) {
        parser.printErrors();
        return 0;
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
    cv::LSD lineDetector(cv::LSD_REFINE_NONE, 0.3, 0.9);

    // Populate with random color codes
    std::vector<cv::Vec3b> color_code = get_color_vector();

    //organized 3D depth image
    Eigen::MatrixXf cloudArrayOrganized(width * height,3);
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);


    int i = startIndex;
    double meanTreatmentTime = 0.0;
    double meanMatTreatmentTime = 0.0;
    double maxTreatTime = 0.0;
    std::cout << "Starting extraction" << std::endl;
    bool runLoop = true;
    while(runLoop) {

        //read images
        rgbImgPath.str("");
        depthImagePath.str("");
        rgbImgPath << dataPath.str() << "rgb_"<< i <<".png";
        depthImagePath << dataPath.str() << "depth_" << i << ".png";
        cv::Mat rgbImage = cv::imread(rgbImgPath.str(), cv::IMREAD_COLOR);
        cv::Mat depthImage = cv::imread(depthImagePath.str(), cv::IMREAD_ANYDEPTH);
        if (!depthImage.data or !rgbImage.data)
            break;
        std::cout << "Read frame " << i << std::endl;

        //set variables
        depthImage.convertTo(depthImage, CV_32F);
        cv::Mat_<uchar> seg_output(height, width, uchar(0));
        vector<Plane_Segment> planeParams;
        vector<Cylinder_Segment> cylinderParams;


        //project depth image in an organized cloud
        double t1 = cv::getTickCount();
        depthOps.get_organized_cloud_array(depthImage, cloudArrayOrganized);
        double time_elapsed = (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
        meanMatTreatmentTime += time_elapsed;


        // Run plane and cylinder detection 
        t1 = cv::getTickCount();
        detector.find_plane_regions(cloudArrayOrganized, planeParams, cylinderParams, seg_output);
        time_elapsed = (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
        meanTreatmentTime += time_elapsed;
        maxTreatTime = max(maxTreatTime, time_elapsed);


        //display 
        double min, max;
        cv::minMaxLoc(depthImage, &min, &max);
        if (min != max){ 
            depthImage -= min;
            depthImage.convertTo(depthImage, CV_8UC1, 255.0/(max-min));
        }

        //display masks on image
        cv::Mat segRgb(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
        detector.apply_masks(rgbImage, color_code, seg_output, planeParams, cylinderParams, segRgb, time_elapsed);


        /*
        //get lines
        std::vector<cv::Vec4f> lines;
        cv::Mat mask = depthImage > 0;

        cv::Mat grayImage;
        cv::cvtColor(rgbImage, grayImage, cv::COLOR_BGR2GRAY);

        lineDetector.detect(grayImage, lines);
        cv::dilate(mask, mask, kernel);
        cv::erode(mask, mask, kernel);
        mask = mask != 0;
        //cv::patchNaNs(mask, 0);

        cv::addWeighted(grayImage, 0.5, mask, 0.5, 0, grayImage);
        cv::cvtColor(grayImage, grayImage, cv::COLOR_GRAY2BGR);

        for(int i = 0; i < lines.size(); i++) {
            cv::Vec4f& pts = lines.at(i);
            cv::Point pt1(pts[0], pts[1]);
            cv::Point pt2(pts[2], pts[3]);
            if (mask.at<uchar>(pt1) == 0  or mask.at<uchar>(pt2) == 0) {
                cv::Point firstQuart = 0.25 * pt1 + 0.75 * pt2;
                cv::Point secQuart = 0.75 * pt1 + 0.25 * pt2;

                //cv::circle(grayImage, firstQuart, 10, cv::Scalar(0, 255, 0));

                //at least a point with depth data
                if (mask.at<uchar>(firstQuart) != 0  or mask.at<uchar>(secQuart) != 0) 
                    cv::line(grayImage, pt1, pt2, cv::Scalar(0, 0, 255), 1);
            }
        }
        //cv::imshow("Seg", grayImage);
*/

        cv::imshow("Seg", segRgb);


        switch(cv::waitKey(1)) {
            //check pressent key
            case 'p': //pause button
                cv::waitKey(-1); //wait until any key is pressed
                break;
            case 'q': //quit button
                runLoop = false;
            default:
                break;
        }
        i++;
    }
    std::cout << "Mean plane treatment time is " << meanTreatmentTime/i << std::endl;
    std::cout << "Mean image shape treatment time is " << meanMatTreatmentTime/i << std::endl;
    std::cout << "max treat time is " << maxTreatTime << std::endl;

    std::cout << "init planes " << detector.resetTime/i << std::endl;
    std::cout << "Init hist " << detector.initTime/i << std::endl;
    std::cout << "grow " << detector.growTime/i << std::endl;
    std::cout << "refine planes " << detector.mergeTime/i << std::endl;
    std::cout << "refine cylinder " << detector.refineTime/i << std::endl;
    std::cout << "setMask " << detector.setMaskTime/i << std::endl;

    cv::destroyAllWindows();
    exit(0);
}










