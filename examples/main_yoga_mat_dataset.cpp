#include <iostream>
#include <fstream>
#include <ctime>
// check file existence
#include <sys/stat.h>

#include <opencv2/opencv.hpp>

#include "RGBD_SLAM.hpp"
#include "Pose.hpp"
#include "parameters.hpp"

#include "utils.hpp"


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

/**
  * \brief checks the existence of a file (from https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-14-17-c)
  */
inline bool is_file_valid (const std::string& fileName) {
    struct stat buffer;
    return (stat (fileName.c_str(), &buffer) == 0);
}

/**
 * \brief Load rgb and depth images from the folder dataPath at index imageIndex
 *
 * \return true is load is successful, else return false
 */
bool load_images(std::stringstream& dataPath, int imageIndex, cv::Mat& rgbImage, cv::Mat& depthImage) 
{
    std::stringstream depthImagePath, rgbImgPath;
    rgbImgPath << dataPath.str() << "rgb_"<< imageIndex <<".png";
    depthImagePath << dataPath.str() << "depth_" << imageIndex << ".png";

    if (is_file_valid(rgbImgPath.str()) and is_file_valid(depthImagePath.str()))
    {

        rgbImage = cv::imread(rgbImgPath.str(), cv::IMREAD_COLOR);
        depthImage = cv::imread(depthImagePath.str(), cv::IMREAD_ANYDEPTH);
        depthImage.convertTo(depthImage, CV_32F);

        //check if images exists
        return depthImage.data and rgbImage.data;
    }
    return false;
}

bool parse_parameters(int argc, char** argv, bool& showPrimitiveMasks, bool& useLineDetection, int& startIndex, unsigned int& jumpImages, bool& shouldSavePoses) 
{
    const cv::String keys = 
        "{help h usage ?  |      | print this message     }"
        "{p primitive     |  1   | display primitive masks }"
        "{l lines         |  0   | Detect lines }"
        "{i index         |  0   | First image to parse   }"
        "{j jump          |  0   | Only take every j image into consideration   }"
        "{s save          |  0   | Should save all the pose to a file }"
        ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("RGBD Slam v0");

    if (parser.has("help")) {
        parser.printMessage();
        return false;
    }

    showPrimitiveMasks = parser.get<bool>("p");
    useLineDetection = parser.get<bool>("l");
    startIndex = parser.get<int>("i");
    jumpImages = parser.get<unsigned int>("j");
    shouldSavePoses = parser.get<bool>("s");

    if(not parser.check()) {
        std::cout << "RGBD SLAM: Some parameters are missing: call with -h to get the list of parameters";
        parser.printErrors();
    }
    return parser.check();
}



int main(int argc, char* argv[]) 
{
    std::stringstream dataPath("../data/yoga/");
    bool showPrimitiveMasks, useLineDetection, shouldSavePoses;
    int startIndex;
    unsigned int jumpFrames = 0;

    if (not parse_parameters(argc, argv, showPrimitiveMasks, useLineDetection, startIndex, jumpFrames, shouldSavePoses)) {
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


    //start with identity pose
    rgbd_slam::Parameters::parse_file("");
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

    rgbd_slam::RGBD_SLAM RGBD_Slam (dataPath, pose, width, height);

    //frame counters
    unsigned int totalFrameTreated = 0;
    unsigned int frameIndex = startIndex;   //current frame index count

    double meanTreatmentTime = 0;

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
        trajectoryFile.open("traj_yoga_mat_" + dateAndTime + ".txt");
        trajectoryFile << "x,y,z,yaw,pitch,roll" << std::endl;
    }

    //stop condition
    bool runLoop = true;
    while(runLoop) {

        if(jumpFrames > 0 and frameIndex % jumpFrames != 0) {
            //do not treat this frame
            ++frameIndex;
            continue;
        }

        // read images
        if(not load_images(dataPath, frameIndex, rgbImage, depthImage))
            break;

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

        if (shouldSavePoses)
        {
            //std::cout << "\x1B[2J\x1B[H" << pose << std::endl;
            const rgbd_slam::vector3& position = pose.get_position();
            const rgbd_slam::quaternion& rotation = pose.get_orientation_quaternion();
            const rgbd_slam::EulerAngles& rotationEuler = rgbd_slam::utils::get_euler_angles_from_quaternion(rotation);
            trajectoryFile << position.x() << "," << position.y() << "," << position.z() << ",";
            trajectoryFile << rotationEuler.yaw << "," << rotationEuler.pitch << "," << rotationEuler.roll << std::endl; 
        }
    }

    if (shouldSavePoses)
        trajectoryFile.close();

    std::cout << std::endl;
    std::cout << "End pose : " << pose << std::endl;
    std::cout << "Process terminated at frame " << frameIndex << std::endl;
    std::cout << std::endl;
    RGBD_Slam.show_statistics(meanTreatmentTime / totalFrameTreated);

    cv::destroyAllWindows();
    exit(0);
}










