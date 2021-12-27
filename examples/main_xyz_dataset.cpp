#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "RGBD_SLAM.hpp"
#include "Pose.hpp"
#include "parameters.hpp"





void check_user_inputs(bool& runLoop, bool& useLineDetection, bool& showPrimitiveMasks) 
{
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
 * returns true is load is successful, else return false
 */

bool parse_parameters(int argc, char** argv, bool& showPrimitiveMasks, bool& useLineDetection, bool& useFrameOdometry, int& startIndex, unsigned int& jumpImages) 
{
    const cv::String keys = 
        "{help h usage ?  |      | print this message     }"
        "{p primitive     |  1   | display primitive masks }"
        "{l lines         |  0   | Detect lines }"
        "{o odometry      |  0   | Use frame odometry }"
        "{i index         |  0   | First image to parse   }"
        "{j jump          |  0   | Only take every j image into consideration   }"
        ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("RGBD SLam v0");

    if (parser.has("help")) {
        parser.printMessage();
        return false;
    }

    showPrimitiveMasks = parser.get<bool>("p");
    useLineDetection = parser.get<bool>("l");
    useFrameOdometry = parser.get<bool>("o");
    startIndex = parser.get<int>("i");
    jumpImages = parser.get<unsigned int>("j");

    if(not parser.check()) {
        std::cout << "RGBD SLAM: Some parameters are missing: call with -h to get the list of parameters";
        parser.printErrors();
    }
    return parser.check();
}

int main(int argc, char* argv[]) 
{
    std::stringstream dataPath("../data/freiburg1_xyz/");
    bool showPrimitiveMasks, useLineDetection, useFrameOdometry;
    int startIndex;
    unsigned int jumpFrames = 0;

    if (not parse_parameters(argc, argv, showPrimitiveMasks, useLineDetection, useFrameOdometry, startIndex, jumpFrames)) {
        return 0;   //could not parse parameters correctly 
    }

    // Get file & folder names
    const std::string rgbImageListPath = dataPath.str() + "rgb.txt";
    const std::string depthImageListPath = dataPath.str() + "depth.txt";

    std::ifstream rgbImagesFile(rgbImageListPath);
    std::ifstream depthImagesFile(depthImageListPath);

    const int width  = 640; 
    const int height = 480;

    rgbd_slam::RGBD_SLAM RGBD_Slam (dataPath, width, height);

    // Load a default set of parameters
    rgbd_slam::Parameters::load_defaut();
    //start with identity pose
    rgbd_slam::utils::Pose pose;
    const vector3 startingPosition(
            rgbd_slam::Parameters::get_starting_position_x(),
            rgbd_slam::Parameters::get_starting_position_y(),
            rgbd_slam::Parameters::get_starting_position_z()
            );
    const quaternion startingRotation;
    pose.set_parameters(startingPosition, startingRotation);


    //frame counters
    unsigned int totalFrameTreated = 0;
    unsigned int frameIndex = startIndex;   //current frame index count

    double meanTreatmentTime = 0;

    //stop condition
    bool runLoop = true;
    for(std::string rgbLine, depthLine; runLoop && std::getline(rgbImagesFile, rgbLine) && std::getline(depthImagesFile, depthLine); ) {

        if (rgbLine[0] == '#' or depthLine[0] == '#')
            continue;
        if(jumpFrames > 0 and frameIndex % jumpFrames != 0) {
            //do not treat this frame
            ++frameIndex;
            continue;
        }

        // Parse lines
        std::istringstream inputRgbString(rgbLine);
        std::istringstream inputDepthString(depthLine);

        double rgbTimeStamp = 0;
        std::string rgbImagePath;
        inputRgbString >> rgbTimeStamp >> rgbImagePath;
        rgbImagePath = dataPath.str() + rgbImagePath;

        double depthTimeStamp = 0;
        std::string depthImagePath;
        inputDepthString >> depthTimeStamp >> depthImagePath;
        depthImagePath = dataPath.str() + depthImagePath;

        // Load images
        cv::Mat rgbImage = cv::imread(rgbImagePath, cv::IMREAD_COLOR);
        cv::Mat depthImage = cv::imread(depthImagePath, cv::IMREAD_GRAYSCALE);

        if (rgbImage.empty() or depthImage.empty())
        {
            std::cerr << "Cannot load " << rgbImagePath << " or " << depthImagePath << std::endl;
            continue;
        }
        // convert to mm & float 32
        depthImage.convertTo(depthImage, CV_32FC1);
        depthImage *= 100;

        // get optimized pose
        double elapsedTime = cv::getTickCount();
        pose = RGBD_Slam.track(rgbImage, depthImage, useLineDetection);
        elapsedTime = (cv::getTickCount() - elapsedTime) / (double)cv::getTickFrequency();
        meanTreatmentTime += elapsedTime;

        // display masks on image
        cv::Mat segRgb = rgbImage.clone();
        RGBD_Slam.get_debug_image(pose, rgbImage, segRgb, elapsedTime, showPrimitiveMasks);
        cv::imshow("RGBD-SLAM", segRgb);

        //check user inputs
        check_user_inputs(runLoop, useLineDetection, showPrimitiveMasks);

        // counters
        ++totalFrameTreated;
        ++frameIndex;
        
        std::cout << "\x1B[2J\x1B[H" << pose << std::endl;
    }

    std::cout << std::endl;
    std::cout << "End pose : " << pose << std::endl;
    std::cout << "Process terminated at frame " << frameIndex << std::endl;
    std::cout << std::endl;
    RGBD_Slam.show_statistics(meanTreatmentTime / totalFrameTreated);

    cv::destroyAllWindows();
    exit(0);
}










