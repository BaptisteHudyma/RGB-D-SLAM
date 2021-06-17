#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor.hpp>


#include "DepthOperations.hpp"
#include "PrimitiveDetection.hpp"
#include "PlaneSegment.hpp"
#include "Parameters.hpp"
#include "MonocularDepthMap.hpp"
#include "DepthMapSegmentation.hpp"
#include "Point_Tracking.hpp"

#include "LineSegmentDetector.hpp"
#include "RGB_Slam.hpp"

#include "Pose.hpp"


#include "GeodesicOperations.hpp"


typedef std::vector<cv::Vec4f> line_vector;
typedef std::list<std::unique_ptr<primitiveDetection::Primitive>> primitive_container; 

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

bool parse_parameters(int argc, char** argv, std::stringstream& dataPath, bool& showPrimitiveMasks, bool& useLineDetection, bool& useFrameOdometry, int& startIndex, unsigned int& jumpImages) {
    const cv::String keys = 
        "{help h usage ?  |      | print this message     }"
        "{f folder        |<none>| folder to parse        }"
        "{p primitive     |  1   | display primitive masks }"
        "{l lines         |  0   | Detect lines }"
        "{o odometry      |  0   | Use frame odometry }"
        "{i index         |  0   | First image to parse   }"
        "{j jump          |  0   | Only take every j image into consideration   }"
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
    jumpImages = parser.get<unsigned int>("j");

    if(not parser.check()) {
        std::cout << "RGBD SLAM: Some paramers are missing: call with -h to get the list of parameters";
        parser.printErrors();
    }
    return parser.check();
}



int main(int argc, char* argv[]) {
    std::stringstream dataPath;
    bool showPrimitiveMasks, useLineDetection, useFrameOdometry;
    bool useDepthSegmentation = false;
    int startIndex;
    unsigned int jumpFrames = 0;

    if (not parse_parameters(argc, argv, dataPath, showPrimitiveMasks, useLineDetection, useFrameOdometry, startIndex, jumpFrames)) {
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
    /*
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
     */

    //plane/cylinder finder
    Primitive_Detection primDetector(height, width, PATCH_SIZE, COS_ANGLE_MAX, MAX_MERGE_DIST, true);
    //Should refine, scale, Gaussian filter sigma
    cv::LSD lineDetector(cv::LSD_REFINE_NONE, 0.3, 0.9);

    // Populate with random color codes
    std::vector<cv::Vec3b> color_code = get_color_vector();

    //organized 3D depth image
    Eigen::MatrixXf cloudArrayOrganized(width * height,3);
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);


    std::vector<cv::Vec3b> colors(150);
    colors[0] = cv::Vec3b(0, 0, 0);//background
    for (int label = 1; label < 150; label++) {
        colors[label] = cv::Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
    }

    Points_Tracking RGB_Slam;

    //start with identity pose
    Pose pose;

    //keep track of the primitives tracked last frame
    primitive_container previousFramePrimitives;


    //frame counters
    unsigned int frameIndex = startIndex;   //current frame index count
    unsigned int totalFrameTreated = 0; 

    //time counters
    double meanTreatmentTime = 0.0;
    double meanMatTreatmentTime = 0.0;
    double meanPoseTreatmentTime = 0.0;
    double maxTreatTime = 0.0;

    double meanLineTreatment = 0.0;

    //stop condition
    bool runLoop = true;
    while(runLoop) {

        if(jumpFrames > 0 and frameIndex % jumpFrames != 0) {
            //do not treat this frame
            ++frameIndex;
            continue;
        }

        //read images
        if(not load_images(dataPath, frameIndex, rgbImage, depthImage))
            break;
        cv::Mat grayImage;
        cv::cvtColor(rgbImage, grayImage, cv::COLOR_BGR2GRAY);

        //set variables
        cv::Mat_<uchar> seg_output(height, width, uchar(0));    //primitive mask mat
        primitive_container primitives;

        //clean warp artefacts
        //cv::Mat newMat;
        //cv::morphologyEx(depthImage, newMat, cv::MORPH_CLOSE, kernel);
        //cv::medianBlur(newMat, newMat, 3);
        //cv::bilateralFilter(newMat, depthImage,  7, 31, 15);

        //project depth image in an organized cloud
        double t1 = cv::getTickCount();
        depthOps.get_organized_cloud_array(depthImage, cloudArrayOrganized);
        double time_elapsed = (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
        meanMatTreatmentTime += time_elapsed;

        // Run primitive detection 
        t1 = cv::getTickCount();
        primDetector.find_primitives(cloudArrayOrganized, primitives, seg_output);
        time_elapsed = (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
        meanTreatmentTime += time_elapsed;
        maxTreatTime = max(maxTreatTime, time_elapsed);


        // this frame points and  assoc
        t1 = cv::getTickCount();

        pose = RGB_Slam.compute_new_pose(rgbImage, depthImage);

        meanPoseTreatmentTime += (cv::getTickCount() - t1) / (double)cv::getTickFrequency();

        //associate primitives
        std::map<int, int> associatedIds;
        if(not previousFramePrimitives.empty()) {
            //find matches between consecutive images
            //compare normals, superposed area (and colors ?)
            for(const std::unique_ptr<Primitive>& prim : primitives) {
                for(const std::unique_ptr<Primitive>& prevPrim : previousFramePrimitives) {
                    if(prim->is_similar(prevPrim)) {
                        associatedIds[prim->get_id()] = prevPrim->get_id();
                        break;
                    }
                }
            }

            //compute pose from matches 
            //-> rotation assuming Manhattan world
            //-> translation with min square minimisation

            //local map reconstruction

            //position refinement from local map

            //global map update from local one

        }
        else {
            //first frame, or no features detected last frame

        }



        //depth map segmentation
        cv::Mat colored;
        if(useDepthSegmentation) {
            colored = rgbImage.clone();
            double reducePourcent = 0.35;
            cv::Mat finalSegmented;
            get_segmented_depth_map(depthImage, finalSegmented, kernel, reducePourcent);
            resize(finalSegmented, finalSegmented, rgbImage.size());
            draw_segmented_labels(finalSegmented, colors, colored);

            //display 
            double min, max;
            cv::minMaxLoc(depthImage, &min, &max);
            if (min != max){ 
                depthImage -= min;
                depthImage.convertTo(depthImage, CV_8UC1, 255.0/(max-min));
            }
        }

        //visual odometry tracking
        /*
           if(useFrameOdometry) {
           Pose estimatedPose = vo.track(grayImage, depthImage);

           if(vo.get_state() == vo.eState_LOST)
           break;
           }
         */

        if(useLineDetection) { //detect lines in image
            t1 = cv::getTickCount();

            //get lines
            line_vector lines;
            cv::Mat mask = depthImage > 0;

            lineDetector.detect(grayImage, lines);

            //fill holes
            cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

            //draw lines with associated depth data
            for(line_vector::size_type i = 0; i < lines.size(); i++) {
                cv::Vec4f& pts = lines.at(i);
                cv::Point pt1(pts[0], pts[1]);
                cv::Point pt2(pts[2], pts[3]);
                if (mask.at<uchar>(pt1) == 0  or mask.at<uchar>(pt2) == 0) {
                    //no depth at extreme points, check first and second quarter
                    cv::Point firstQuart = 0.25 * pt1 + 0.75 * pt2;
                    cv::Point secQuart = 0.75 * pt1 + 0.25 * pt2;

                    //at least a point with depth data
                    if (mask.at<uchar>(firstQuart) != 0  or mask.at<uchar>(secQuart) != 0) 
                        cv::line(rgbImage, pt1, pt2, cv::Scalar(0, 0, 255), 1);
                    else    //no depth data
                        cv::line(rgbImage, pt1, pt2, cv::Scalar(255, 0, 255), 1);
                }
                else
                    //line with associated depth
                    cv::line(rgbImage, pt1, pt2, cv::Scalar(0, 255, 255), 1);

            }
            meanLineTreatment += (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
        }

        //display masks on image
        cv::Mat segRgb = rgbImage.clone();//(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

        if(showPrimitiveMasks)
            primDetector.apply_masks(rgbImage, color_code, seg_output, primitives, segRgb, associatedIds, time_elapsed);

        // exchange frames features
        previousFramePrimitives.swap(primitives);


        //display with mono mask
        //cv::cvtColor(depthImage, depthImage, cv::COLOR_GRAY2BGR);
        if(useDepthSegmentation)
            cv::hconcat(segRgb, colored, segRgb);
        cv::imshow("RGBD-SLAM", segRgb);

        check_user_inputs(runLoop, useLineDetection, showPrimitiveMasks);


        //counters
        ++frameIndex;
        ++totalFrameTreated;
    }
    std::cout << "Process terminated at frame " << frameIndex << std::endl;
    std::cout << "Mean image to point cloud treatment time is " << meanMatTreatmentTime / totalFrameTreated << std::endl;
    std::cout << "Mean plane treatment time is " << meanTreatmentTime / totalFrameTreated << std::endl;
    std::cout << "max treat time is " << maxTreatTime << std::endl;
    std::cout << std::endl;
    if(useLineDetection)
        std::cout << "Mean line detection time is " << meanLineTreatment / totalFrameTreated << std::endl;
    std::cout << "Mean pose estimation time is " << meanPoseTreatmentTime / totalFrameTreated << std::endl;

    //std::cout << "init planes " << primDetector.resetTime/i << std::endl;
    //std::cout << "Init hist " << primDetector.initTime/i << std::endl;
    //std::cout << "grow " << primDetector.growTime/i << std::endl;
    //std::cout << "refine planes " << primDetector.mergeTime/i << std::endl;
    //std::cout << "refine cylinder " << primDetector.refineTime/i << std::endl;
    //std::cout << "setMask " << primDetector.setMaskTime/i << std::endl;

    cv::destroyAllWindows();
    exit(0);
}










