// Dataset available from "Fast Cylinder and Plane Extraction from Depth Cameras for Visual Odometry"

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

void check_user_inputs(bool& shouldStop, bool& shouldUseLineDetection, bool& shouldDisplayPrimitiveMasks)
{
    switch (cv::waitKey(1))
    {
        // check pressed key
        case 'l':
            shouldUseLineDetection = not shouldUseLineDetection;
            break;
        case 's':
            shouldDisplayPrimitiveMasks = not shouldDisplayPrimitiveMasks;
            break;
        case 'p':            // pause button
            cv::waitKey(-1); // wait until any key is pressed
            break;
        case 'q': // quit button
            shouldStop = false;
        default:
            break;
    }
}

/**
 * \brief checks the existence of a file (from
 * https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-14-17-c)
 */
inline bool is_file_valid(const std::string& fileName)
{
    struct stat buffer;
    return (stat(fileName.c_str(), &buffer) == 0);
}

/**
 * \brief Load rgb and depth images from the folder dataPath at index imageIndex
 *
 * \return true is load is successful, else return false
 */
bool load_images(const std::stringstream& dataPath, const uint imageIndex, cv::Mat& rgbImage, cv::Mat& depthImage)
{
    std::stringstream depthImagePath;
    std::stringstream rgbImgPath;
    rgbImgPath << dataPath.str() << "rgb_" << imageIndex << ".png";
    depthImagePath << dataPath.str() << "depth_" << imageIndex << ".png";

    if (is_file_valid(rgbImgPath.str()) and is_file_valid(depthImagePath.str()))
    {
        rgbImage = cv::imread(rgbImgPath.str(), cv::IMREAD_COLOR);
        depthImage = cv::imread(depthImagePath.str(), cv::IMREAD_ANYDEPTH);
        depthImage.convertTo(depthImage, CV_32F);

        // check if images exists
        return depthImage.data and rgbImage.data;
    }
    return false;
}

bool parse_parameters(int argc,
                      char** argv,
                      std::string& dataset,
                      bool& shouldDisplayPrimitiveMasks,
                      bool& shouldDisplayStagedPoints,
                      bool& shouldUseLineDetection,
                      int& startIndex,
                      unsigned int& jumpImages,
                      unsigned int& fpsTarget,
                      bool& shouldSavePoses)
{
    const cv::String keys =
            "{help h usage ?  |      | print this message     }"
            "{@dataset        | yoga | Dataset to process }"
            "{p primitive     |  1   | display primitive masks }"
            "{d staged        |  0   | display points in staged container }"
            "{l lines         |  0   | Detect lines }"
            "{i index         |  0   | First image to parse   }"
            "{j jump          |  0   | Only take every j image into consideration   }"
            "{r fps           |  30  | Used to slow down the treatment to correspond to a certain frame rate }"
            "{s save          |  0   | Should save all the pose to a file }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("RGBD Slam v0");

    if (parser.has("help"))
    {
        parser.printMessage();
        return false;
    }

    dataset = parser.get<std::string>("@dataset");
    shouldDisplayPrimitiveMasks = parser.get<bool>("p");
    shouldDisplayStagedPoints = parser.get<bool>("d");
    shouldUseLineDetection = parser.get<bool>("l");
    startIndex = parser.get<int>("i");
    jumpImages = parser.get<unsigned int>("j");
    fpsTarget = parser.get<unsigned int>("r");
    shouldSavePoses = parser.get<bool>("s");

    if (not parser.check())
    {
        std::cout << "RGBD SLAM: Some parameters are missing: call with -h to get the list of parameters";
        parser.printErrors();
    }
    return parser.check();
}

int main(int argc, char* argv[])
{
    std::string dataset;
    bool shouldDisplayPrimitiveMasks;
    bool shouldDisplayStagedPoints;
    bool shouldUseLineDetection;
    bool shouldSavePoses;
    int startIndex;
    uint jumpFrames = 0;
    uint fpsTarget;

    if (not parse_parameters(argc,
                             argv,
                             dataset,
                             shouldDisplayPrimitiveMasks,
                             shouldDisplayStagedPoints,
                             shouldUseLineDetection,
                             startIndex,
                             jumpFrames,
                             fpsTarget,
                             shouldSavePoses))
    {
        return 0; // could not parse parameters correctly
    }
    const std::stringstream dataPath("./data/CAPE/" + dataset + "/");

    // Load a default set of parameters
    const bool isParsingSuccesfull = rgbd_slam::Parameters::parse_file(dataPath.str() + "configuration.yaml");
    if (not isParsingSuccesfull)
    {
        std::cout << "Could not parse the parameter file at  " << (dataPath.str() + "configuration.yaml") << std::endl;
        return -1;
    }

    const uint width = rgbd_slam::Parameters::get_camera_1_image_size().x();  // 640
    const uint height = rgbd_slam::Parameters::get_camera_1_image_size().y(); // 480

    // start with ground truth pose
    rgbd_slam::utils::Pose pose;

    rgbd_slam::RGBD_SLAM RGBD_Slam(pose, width, height);

    // frame counters
    unsigned int totalFrameTreated = 0;
    unsigned int frameIndex = startIndex; // current frame index count

    double meanTreatmentDuration = 0;

    std::ofstream trajectoryFile;
    if (shouldSavePoses)
    {
        std::time_t timeOfTheDay = std::time(0);
        std::tm* gmtTime = std::gmtime(&timeOfTheDay);
        std::string dateAndTime = std::to_string(1900 + gmtTime->tm_year) + "-" + std::to_string(1 + gmtTime->tm_mon) +
                                  "-" + std::to_string(gmtTime->tm_mday) + "_" + std::to_string(1 + gmtTime->tm_hour) +
                                  ":" + std::to_string(1 + gmtTime->tm_min) + ":" + std::to_string(1 + gmtTime->tm_sec);
        std::cout << dateAndTime << std::endl;
        trajectoryFile.open("traj_" + dataset + "_mat_" + dateAndTime + ".txt");
        trajectoryFile << "x,y,z,yaw,pitch,roll" << std::endl;
    }

    // stop condition
    bool shouldStop = true;
    while (shouldStop)
    {
        if (jumpFrames > 0 and frameIndex % jumpFrames != 0)
        {
            // do not treat this frame
            ++frameIndex;
            continue;
        }

        // read images
        cv::Mat rgbImage;
        cv::Mat_<float> depthImage;
        if (not load_images(dataPath, frameIndex, rgbImage, depthImage))
            break;
        assert(static_cast<uint>(rgbImage.cols) == width and static_cast<uint>(rgbImage.rows) == height);
        assert(static_cast<uint>(depthImage.cols) == width and static_cast<uint>(depthImage.rows) == height);

        // rectify the depth image before next step
        RGBD_Slam.rectify_depth(depthImage);

        // get optimized pose
        const double trackingStartTime = static_cast<double>(cv::getTickCount());
        pose = RGBD_Slam.track(rgbImage, depthImage, shouldUseLineDetection);
        const double trackingDuration =
                (static_cast<double>(cv::getTickCount()) - trackingStartTime) / cv::getTickFrequency();
        meanTreatmentDuration += trackingDuration;

        // display masks on image
        const cv::Mat& segRgb = RGBD_Slam.get_debug_image(
                pose, rgbImage, trackingDuration, shouldDisplayStagedPoints, shouldDisplayPrimitiveMasks);
        cv::imshow("RGBD-SLAM", segRgb);

        // check user inputs
        check_user_inputs(shouldStop, shouldUseLineDetection, shouldDisplayPrimitiveMasks);

        // counters
        ++totalFrameTreated;
        ++frameIndex;

        if (shouldSavePoses)
        {
            // std::cout << "\x1B[2J\x1B[H" << pose << std::endl;
            const rgbd_slam::vector3& position = pose.get_position();
            const rgbd_slam::quaternion& rotation = pose.get_orientation_quaternion();
            const rgbd_slam::EulerAngles& rotationEuler = rgbd_slam::utils::get_euler_angles_from_quaternion(rotation);
            trajectoryFile << position.x() << "," << position.y() << "," << position.z() << ",";
            trajectoryFile << rotationEuler.yaw << "," << rotationEuler.pitch << "," << rotationEuler.roll << std::endl;
        }

        // wait to adjust framerate
        if (trackingDuration < 1.0 / fpsTarget)
        {
            usleep((1.0 / static_cast<double>(fpsTarget) - trackingDuration) * 1e6);
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
