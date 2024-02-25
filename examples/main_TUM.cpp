// The dataset can be found here:
// https://vision.in.tum.de/data/datasets/rgbd-dataset

#include <iostream>
#include <fstream>
#include <ctime>
// check file existence
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sys/stat.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include "logger.hpp"
#include "rgbd_slam.hpp"
#include "pose.hpp"
#include "parameters.hpp"
#include "angle_utils.hpp"
#include "types.hpp"
#include "TUM_parser.hpp"

void check_user_inputs(bool& shouldStop)
{
    switch (cv::waitKey(1))
    {
        // check pressed key
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

bool parse_parameters(int argc,
                      char** argv,
                      std::string& dataset,
                      bool& shouldDisplayStagedFeatures,
                      int& startIndex,
                      unsigned int& jumpImages,
                      unsigned int& fpsTarget,
                      bool& shouldSavePoses)
{
    const cv::String keys =
            "{help h usage ?  |      | print this message     }"
            "{@dataset        | yoga | Dataset to process }"
            "{d staged        |  0   | display features in staged container }"
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
    shouldDisplayStagedFeatures = parser.get<bool>("d");
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

rgbd_slam::utils::Pose get_ground_truth(const std::string& groundTruthLine)
{
    assert(groundTruthLine != "");

    // parse ground truth
    std::istringstream inputGroundTruth(groundTruthLine);
    double timestamp = 0;
    rgbd_slam::vector3 groundTruthPosition;
    rgbd_slam::quaternion groundTruthRotation;
    inputGroundTruth >> timestamp >> groundTruthPosition.x() >> groundTruthPosition.y() >> groundTruthPosition.z() >>
            groundTruthRotation.x() >> groundTruthRotation.y() >> groundTruthRotation.z() >> groundTruthRotation.w();

    return rgbd_slam::utils::Pose(groundTruthPosition * 1000.0, groundTruthRotation);
}

std::vector<Data> get_data_association(const std::string& dataPath)
{
    const std::string groundTruthPath = dataPath + "groundtruth.txt";
    std::ifstream associationFile(dataPath + "associations.txt");
    if (associationFile.is_open())
    {
        // an association file already exists
        std::cout << "Using association file" << std::endl;

        return DatasetParser::parse_association_file(dataPath, groundTruthPath);
    }
    else
    {
        // parse the folder myself...
        std::cout << "Generate association data" << std::endl;

        // Get file & folder names
        const std::string rgbImageListPath = dataPath + "rgb.txt";
        const std::string depthImageListPath = dataPath + "depth.txt";

        return DatasetParser::parse_dataset(rgbImageListPath, depthImageListPath, groundTruthPath);
    }
}

int main(int argc, char* argv[])
{
    std::string dataset;
    bool shouldDisplayStagedFeatures;
    bool shouldSavePoses;
    int startIndex;
    uint jumpFrames = 0;
    uint fpsTarget;

    if (not parse_parameters(argc,
                             argv,
                             dataset,
shouldDisplayStagedFeatures,
                             startIndex,
                             jumpFrames,
                             fpsTarget,
                             shouldSavePoses))
    {
        return 0; // could not parse parameters correctly
    }
    const std::stringstream dataPath("./data/TUM/" + dataset + "/");

    const std::vector<Data>& datasetContainer = get_data_association(dataPath.str());

    if (datasetContainer.empty())
    {
        std::cout << "Could not load any dataset elements at " << dataPath.str() << std::endl;
        return -1;
    }

    rgbd_slam::utils::Pose pose;
    if (const GroundTruth& initialGroundTruth = datasetContainer[0].groundTruth; initialGroundTruth.isValid)
    {
        pose.set_parameters(initialGroundTruth.position, initialGroundTruth.rotation);
    }

    // Load a default set of parameters
    const bool isParsingSuccesfull = rgbd_slam::Parameters::parse_file(dataPath.str() + "configuration.yaml");
    if (not isParsingSuccesfull)
    {
        std::cout << "Could not parse the parameter file at  " << (dataPath.str() + "configuration.yaml") << std::endl;
        return -1;
    }

    const uint width = rgbd_slam::Parameters::get_camera_1_image_size().x();  // 640
    const uint height = rgbd_slam::Parameters::get_camera_1_image_size().y(); // 480

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
        trajectoryFile.open("traj_TUM_" + dataset + "_" + dateAndTime + ".txt");
        trajectoryFile << "x,y,z,yaw,pitch,roll" << std::endl;
    }

    double positionError = 0;
    double rotationError = 0;

    // stop condition
    bool shouldStop = true;
    bool isGroundTruthAvailable = false;
    for (const Data& imageData: datasetContainer)
    {
        // out condition
        if (not shouldStop)
            break;

        if (jumpFrames > 0 and frameIndex % jumpFrames != 0)
        {
            // do not treat this frame
            ++frameIndex;
            continue;
        }

        const std::string rgbImagePath = dataPath.str() + imageData.rgbImage.imagePath;
        const std::string depthImagePath = dataPath.str() + imageData.depthImage.imagePath;

        // Load images
        cv::Mat rgbImage = cv::imread(rgbImagePath, cv::IMREAD_COLOR);
        cv::Mat depthImage = cv::imread(depthImagePath, cv::IMREAD_ANYDEPTH);

        if (rgbImage.empty())
        {
            rgbd_slam::outputs::log("No color image input, use black image");
            if (imageData.rgbImage.isValid)
                std::cerr << "Cannot load rgb image " << rgbImagePath << std::endl;
            rgbImage = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
        }
        if (depthImage.empty())
        {
            rgbd_slam::outputs::log("No depth image input, use empty depth");
            if (imageData.depthImage.isValid)
                std::cerr << "Could not load depth image " << depthImagePath << std::endl;

            depthImage = cv::Mat(height, width, CV_16UC1, cv::Scalar(0.0));
        }
        assert(static_cast<uint>(rgbImage.cols) == width and static_cast<uint>(rgbImage.rows) == height);
        assert(static_cast<uint>(depthImage.cols) == width and static_cast<uint>(depthImage.rows) == height);

        // convert to mm & float 32
        depthImage.convertTo(depthImage, CV_32FC1, 1.0 / 5.0);

        // clean warp artefacts
#if 0
        cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
        cv::Mat newMat;
        cv::morphologyEx(depthImage, newMat, cv::MORPH_CLOSE, kernel);
        cv::medianBlur(newMat, newMat, 3);
        cv::bilateralFilter(newMat, depthImage,  7, 31, 15);
#endif

        // rectify the depth image before next step (already rectified in TU%M datasets)
        // RGBD_Slam.rectify_depth(depthImage);

        // get optimized pose
        const double trackingStartTime = static_cast<double>(cv::getTickCount());
        pose = RGBD_Slam.track(rgbImage, depthImage);
        const double trackingDuration =
                (static_cast<double>(cv::getTickCount()) - trackingStartTime) / (double)cv::getTickFrequency();
        meanTreatmentDuration += trackingDuration;

        // estimate error to ground truth
        isGroundTruthAvailable = imageData.groundTruth.isValid;
        if (isGroundTruthAvailable)
        {
            rgbd_slam::utils::PoseBase groundTruthPose(imageData.groundTruth.position, imageData.groundTruth.rotation);
            positionError = pose.get_position_error(groundTruthPose);
            rotationError = pose.get_rotation_error(groundTruthPose);
        }

        // display masks on image
        const cv::Mat& segRgb = RGBD_Slam.get_debug_image(pose,
                                                          rgbImage,
                                                          trackingDuration,
                                                          shouldDisplayStagedFeatures);
        cv::imshow("RGBD-SLAM", segRgb);

        // check user inputs
        check_user_inputs(shouldStop);

        // counters
        ++totalFrameTreated;
        ++frameIndex;

        if (shouldSavePoses)
        {
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
    if (isGroundTruthAvailable)
        std::cout << "Pose error: " << positionError / 10.0 << " cm | " << rotationError << " °" << std::endl;
    std::cout << "End pose : " << pose << std::endl;
    std::cout << "Process terminated at frame " << frameIndex << std::endl;
    std::cout << std::endl;
    RGBD_Slam.show_statistics(meanTreatmentDuration / totalFrameTreated);

    cv::destroyAllWindows();
    return 0;
}
