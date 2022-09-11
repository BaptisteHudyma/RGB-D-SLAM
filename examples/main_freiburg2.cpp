// The dataset can be found here:
// https://vision.in.tum.de/data/datasets/rgbd-dataset



#include <iostream>
#include <fstream>
#include <ctime>
// check file existence
#include <sys/stat.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include "rgbd_slam.hpp"
#include "pose.hpp"
#include "parameters.hpp"
#include "angle_utils.hpp"


void check_user_inputs(bool& shouldRunLoop, bool& useLineDetection, bool& showPrimitiveMasks) 
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
            shouldRunLoop = false;
        default:
            break;
    }
}

/**
 * \brief checks the existence of a file (from https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-14-17-c)
 */
inline bool is_file_valid (const std::string& fileName) {
    struct stat buffer;
    return (stat (fileName.c_str(), &buffer) == 0);
}


bool parse_parameters(int argc, char** argv, std::string& dataset, bool& showPrimitiveMasks, bool& showStagedPoints, bool& useLineDetection, int& startIndex, unsigned int& jumpImages, unsigned int& fpsTarget, bool& shouldSavePoses) 
{
    const cv::String keys = 
        "{help h usage ?  |                  | print this message     }"
        "{@dataset        | desk_with_person | The dataset to read}"
        "{p primitive     |  1               | display primitive masks }"
        "{d staged        |  0               | display points in staged container }"
        "{l lines         |  0               | Detect lines }"
        "{i index         |  0               | First image to parse   }"
        "{j jump          |  0               | Only take every j image into consideration   }"
        "{r fps           |  30              | Used to slow down the treatment to correspond to a certain frame rate }"
        "{s save          |  0               | Should save all the pose to a file }"
        ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("RGBD Slam v0");

    if (parser.has("help")) {
        parser.printMessage();
        return false;
    }
    
    dataset = parser.get<std::string>("@dataset");
    showPrimitiveMasks = parser.get<bool>("p");
    showStagedPoints = parser.get<bool>("d");
    useLineDetection = parser.get<bool>("l");
    startIndex = parser.get<int>("i");
    jumpImages = parser.get<unsigned int>("j");
    fpsTarget = parser.get<unsigned int>("r");
    shouldSavePoses = parser.get<bool>("s");

    if(not parser.check()) {
        std::cout << "RGBD SLAM: Some parameters are missing: call with -h to get the list of parameters";
        parser.printErrors();
    }
    return parser.check();
}

int main(int argc, char* argv[]) 
{
    std::string dataset;
    bool showPrimitiveMasks, showStagedPoints, useLineDetection, shouldSavePoses;
    int startIndex;
    unsigned int jumpFrames = 0, fpsTarget;

    if (not parse_parameters(argc, argv, dataset, showPrimitiveMasks, showStagedPoints, useLineDetection, startIndex, jumpFrames, fpsTarget, shouldSavePoses)) {
        return 0;   //could not parse parameters correctly 
    }
    const std::stringstream dataPath("./data/freiburg2_" + dataset + "/");

    // Get file & folder names
    const std::string rgbImageListPath = dataPath.str() + "rgb.txt";
    const std::string depthImageListPath = dataPath.str() + "depth.txt";

    std::ifstream rgbImagesFile(rgbImageListPath);
    std::ifstream depthImagesFile(depthImageListPath);

    const int width  = 640; 
    const int height = 480;


    // Load a default set of parameters
    rgbd_slam::Parameters::parse_file(dataPath.str() + "configuration.yaml");
    //start with identity pose
    rgbd_slam::utils::Pose pose;
    const rgbd_slam::vector3 startingPosition(
            rgbd_slam::Parameters::get_starting_position_x(),
            rgbd_slam::Parameters::get_starting_position_y(),
            rgbd_slam::Parameters::get_starting_position_z()
            );
    const rgbd_slam::EulerAngles startingRotationEuler(
            rgbd_slam::Parameters::get_starting_rotation_x(),
            rgbd_slam::Parameters::get_starting_rotation_y(),
            rgbd_slam::Parameters::get_starting_rotation_z()
            );
    const rgbd_slam::quaternion& startingRotation = rgbd_slam::utils::get_quaternion_from_euler_angles(startingRotationEuler);
    pose.set_parameters(startingPosition, startingRotation);

    rgbd_slam::RGBD_SLAM RGBD_Slam (pose, width, height);

    //frame counters
    unsigned int totalFrameTreated = 0;
    unsigned int frameIndex = startIndex;   //current frame index count

    double meanTreatmentDuration = 0;

    std::ofstream trajectoryFile;
    if (shouldSavePoses)
    {
        std::time_t timeOfTheDay = std::time(0);
        std::tm* gmtTime = std::gmtime(&timeOfTheDay);
        std::string dateAndTime = 
            std::to_string(1900 + gmtTime->tm_year) + "-" + 
            std::to_string(1 + gmtTime->tm_mon) + "-" + 
            std::to_string(gmtTime->tm_mday) + "_" + 
            std::to_string(1 + gmtTime->tm_hour) + ":" + 
            std::to_string(1 + gmtTime->tm_min) + ":" +
            std::to_string(1 + gmtTime->tm_sec);
        std::cout << dateAndTime << std::endl;
        trajectoryFile.open("traj_freiburg1_" + dataset + "_" + dateAndTime + ".txt");
        trajectoryFile << "x,y,z,yaw,pitch,roll" << std::endl;
    }

    //stop condition
    bool shouldRunLoop = true;
    for(std::string rgbLine, depthLine; shouldRunLoop and std::getline(rgbImagesFile, rgbLine) and std::getline(depthImagesFile, depthLine); ) 
    {
        // skip comments    
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
        rgbImagePath.insert(0, dataPath.str());

        double depthTimeStamp = 0;
        std::string depthImagePath;
        inputDepthString >> depthTimeStamp >> depthImagePath;
        depthImagePath.insert(0, dataPath.str());

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
        depthImage *= 100.0 / 5.0;

        //clean warp artefacts
#if 0
        cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
        cv::Mat newMat;
        cv::morphologyEx(depthImage, newMat, cv::MORPH_CLOSE, kernel);
        cv::medianBlur(newMat, newMat, 3);
        cv::bilateralFilter(newMat, depthImage,  7, 31, 15);
#endif

        // get optimized pose
        const double trackingStartTime = cv::getTickCount();
        pose = RGBD_Slam.track(rgbImage, depthImage, useLineDetection);
        const double trackingDuration = (cv::getTickCount() - trackingStartTime) / (double)cv::getTickFrequency();
        meanTreatmentDuration += trackingDuration;

        // display masks on image
        cv::Mat segRgb = rgbImage.clone();
        RGBD_Slam.get_debug_image(pose, rgbImage, segRgb, trackingDuration, showStagedPoints, showPrimitiveMasks);
        cv::imshow("RGBD-SLAM", segRgb);

        //check user inputs
        check_user_inputs(shouldRunLoop, useLineDetection, showPrimitiveMasks);

        // counters
        ++totalFrameTreated;
        ++frameIndex;

        if (shouldSavePoses)
        {
            //std::cout << "\x1B[2J\x1B[H" << pose << std::endl;
            const rgbd_slam::vector3& position = pose.get_position();
            const rgbd_slam::quaternion& rotation = pose.get_orientation_quaternion();
            const rgbd_slam::EulerAngles& rotationEuler = rgbd_slam::utils::get_euler_angles_from_quaternion(rotation);
            trajectoryFile << position.x() << "," << position.y() << "," << position.z() << ",";
            trajectoryFile << rotationEuler.yaw << "," << rotationEuler.pitch << "," << rotationEuler.roll << std::endl; 
        }

        //wait to adjust framerate
        if (trackingDuration < 1.0 / fpsTarget)
        {
            usleep( (1.0 / static_cast<double>(fpsTarget) - trackingDuration) * 1e6);
        }
    }

    if (shouldSavePoses)
        trajectoryFile.close();

    std::cout << std::endl;
    std::cout << "End pose : " << pose << std::endl;
    std::cout << "Process terminated at frame " << frameIndex << std::endl;
    std::cout << std::endl;
    RGBD_Slam.show_statistics(meanTreatmentDuration / totalFrameTreated);

    cv::destroyAllWindows();
    return 0;
}










