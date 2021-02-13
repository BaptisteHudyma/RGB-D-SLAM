#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen/Dense>


#include "DepthOperations.hpp"
#include "PrimitiveDetection.hpp"
#include "PlaneSegment.hpp"
#include "Parameters.hpp"
#include "MonocularDepthMap.hpp"

#include "LineSegmentDetector.hpp"
#include "RGB_Slam.hpp"

using namespace primitiveDetection;
using namespace poseEstimation;
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

void check_user_inputs(bool& runLoop, bool& useLineDetection, bool& showPrimitiveMasks) {
    switch(cv::waitKey(1)) {
        //check pressed key
        case 'l':
            useLineDetection = not useLineDetection;
            break;
        case 's':
            showPrimitiveMasks = not showPrimitiveMasks;
            break;
        case 'p': //pause button
            cv::waitKey(-1); //wait until any key is pressed
            break;
        case 'q': //quit button
            runLoop = false;
        default:
            break;
    }
}

/*
 * Load rgb and depth images from the folder dataPath at index imageIndex
 *
 *  returns true is load is successful, else return false
 */
bool load_images(std::stringstream& dataPath, int imageIndex, cv::Mat& rgbImage, cv::Mat& depthImage) {
    std::stringstream depthImagePath, rgbImgPath;
    rgbImgPath << dataPath.str() << "rgb_"<< imageIndex <<".png";
    depthImagePath << dataPath.str() << "depth_" << imageIndex << ".png";

    rgbImage = cv::imread(rgbImgPath.str(), cv::IMREAD_COLOR);
    depthImage = cv::imread(depthImagePath.str(), cv::IMREAD_ANYDEPTH);
    depthImage.convertTo(depthImage, CV_32F);

    //check if images exists
    return depthImage.data and rgbImage.data;
}

bool parse_parameters(int argc, char** argv, std::stringstream& dataPath, bool& showPrimitiveMasks, bool& useLineDetection, bool& useFrameOdometry, int& startIndex) {
    const cv::String keys = 
        "{help h usage ?  |      | print this message     }"
        "{f folder        |<none>| folder to parse        }"
        "{p primitive     |  1   | display primitive masks }"
        "{l lines         |  0   | Detect lines }"
        "{o odometry      |  0   | Use frame odometry }"
        "{i index         |  0   | First image to parse   }"
        ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Plane Detection v1");

    if (parser.has("help")) {
        parser.printMessage();
        return false;
    }

    dataPath << parser.get<cv::String>("f") << "/";
    showPrimitiveMasks = parser.get<bool>("p");
    useLineDetection = parser.get<bool>("l");
    useFrameOdometry = parser.get<bool>("o");
    startIndex = parser.get<int>("i");

    if(not parser.check()) {
        parser.printErrors();
    }
    return parser.check();
}

int main(int argc, char* argv[]) {
    std::stringstream dataPath;
    bool showPrimitiveMasks, useLineDetection, useFrameOdometry;
    int startIndex;

    if (not parse_parameters(argc, argv, dataPath, showPrimitiveMasks, useLineDetection, useFrameOdometry, startIndex)) {
        return 0;   //could not parse parameters correctly 
    }

    int width, height;
    cv::Mat rgbImage, depthImage;
    if(not load_images(dataPath, startIndex, rgbImage, depthImage) ) {
        std::cout << "Error loading images at " << dataPath.str() << std::endl;
        return -1;
    }

    width = rgbImage.cols;
    height = rgbImage.rows;

    // Get intrinsics parameters
    std::stringstream calibPath, calibYAMLPath;
    calibPath << dataPath.str() << "calib_params.xml";
    calibYAMLPath << dataPath.str() << "calib_params.yaml";

    //primitive connected graph creator
    Depth_Operations depthOps(calibPath.str(), width, height, PATCH_SIZE);
    if (not depthOps.is_ok()) {
        return -1;
    }

    //visual odometry params
    Parameters params;
    if (not params.init_from_file(calibYAMLPath.str())) {
        std::cout << "Failed to load YAML param file at: " << calibYAMLPath.str() << std::endl;
        return -1;
    }

    params.set_fx(depthOps.get_rgb_fx());
    params.set_fy(depthOps.get_rgb_fy());
    params.set_cx(depthOps.get_rgb_cx());
    params.set_cy(depthOps.get_rgb_cy());

    params.set_height(height);
    params.set_width(width);

    //visual odom class
    RGB_SLAM vo(params);


    //plane/cylinder finder
    Primitive_Detection primDetector(height, width, PATCH_SIZE, COS_ANGLE_MAX, MAX_MERGE_DIST, true);
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
    bool runLoop = true;
    while(runLoop) {

        //read images
        if(not load_images(dataPath, i, rgbImage, depthImage))
            break;
        std::cout << "Read frame " << i << std::endl;
        cv::Mat grayImage;
        cv::cvtColor(rgbImage, grayImage, cv::COLOR_BGR2GRAY);

        //set variables
        cv::Mat_<uchar> seg_output(height, width, uchar(0));    //primitive mask mat
        vector<Plane_Segment> planeParams;
        vector<Cylinder_Segment> cylinderParams;

        //project depth image in an organized cloud
        double t1 = cv::getTickCount();
        depthOps.get_organized_cloud_array(depthImage, cloudArrayOrganized);
        double time_elapsed = (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
        meanMatTreatmentTime += time_elapsed;

        // Run primitive detection 
        t1 = cv::getTickCount();
        primDetector.find_primitives(cloudArrayOrganized, planeParams, cylinderParams, seg_output);
        time_elapsed = (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
        meanTreatmentTime += time_elapsed;
        maxTreatTime = max(maxTreatTime, time_elapsed);


        //visual odometry tracking
        if(useFrameOdometry) {
            Pose estimatedPose = vo.track(grayImage, depthImage);

            if(vo.get_state() == vo.eState_LOST)
                break;
        }

        //display 
        double min, max;
        cv::minMaxLoc(depthImage, &min, &max);
        if (min != max){ 
            depthImage -= min;
            depthImage.convertTo(depthImage, CV_8UC1, 255.0/(max-min));
        }


        if(useLineDetection) { //detect lines in image
            //get lines
            std::vector<cv::Vec4f> lines;
            cv::Mat mask = depthImage > 0;

            lineDetector.detect(grayImage, lines);

            //fill holes
            cv::dilate(mask, mask, kernel);
            cv::erode(mask, mask, kernel);

            for(int i = 0; i < lines.size(); i++) {
                cv::Vec4f& pts = lines.at(i);
                cv::Point pt1(pts[0], pts[1]);
                cv::Point pt2(pts[2], pts[3]);
                if (mask.at<uchar>(pt1) == 0  or mask.at<uchar>(pt2) == 0) {
                    cv::Point firstQuart = 0.25 * pt1 + 0.75 * pt2;
                    cv::Point secQuart = 0.75 * pt1 + 0.25 * pt2;

                    //at least a point with depth data
                    if (mask.at<uchar>(firstQuart) != 0  or mask.at<uchar>(secQuart) != 0) 
                        cv::line(rgbImage, pt1, pt2, cv::Scalar(0, 0, 255), 1);
                }
            }
        }

        //display masks on image
        cv::Mat segRgb = rgbImage.clone();//(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

        if(showPrimitiveMasks)
            primDetector.apply_masks(rgbImage, color_code, seg_output, planeParams, cylinderParams, segRgb, time_elapsed);

        //display with mono mask
        //cv::applyColorMap(depthImage, depthImage, cv::COLORMAP_PLASMA);
        //hconcat(depthImage, segRgb, segRgb);
        cv::imshow("Seg", segRgb);

        check_user_inputs(runLoop, useLineDetection, showPrimitiveMasks);
        i++;
    }
    std::cout << "Mean plane treatment time is " << meanTreatmentTime/i << std::endl;
    std::cout << "Mean image shape treatment time is " << meanMatTreatmentTime/i << std::endl;
    std::cout << "max treat time is " << maxTreatTime << std::endl;

    //std::cout << "init planes " << primDetector.resetTime/i << std::endl;
    //std::cout << "Init hist " << primDetector.initTime/i << std::endl;
    //std::cout << "grow " << primDetector.growTime/i << std::endl;
    //std::cout << "refine planes " << primDetector.mergeTime/i << std::endl;
    //std::cout << "refine cylinder " << primDetector.refineTime/i << std::endl;
    //std::cout << "setMask " << primDetector.setMaskTime/i << std::endl;

    cv::destroyAllWindows();
    exit(0);
}










