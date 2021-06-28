#include <iostream>

#include <opencv2/opencv.hpp>

#include "RGBD_SLAM.hpp"
#include "Pose.hpp"





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

    primitiveDetection::RGBD_SLAM RGBD_Slam (dataPath, width, height, 600);

    //start with identity pose
    poseEstimation::Pose pose;


    //frame counters
    unsigned int frameIndex = startIndex;   //current frame index count

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

        // display masks on image
        cv::Mat segRgb = rgbImage.clone();
        RGBD_Slam.get_debug_image(rgbImage, segRgb, elapsedTime, showPrimitiveMasks);
        cv::imshow("RGBD-SLAM", segRgb);

        //check user inputs
        check_user_inputs(runLoop, useLineDetection, showPrimitiveMasks);

        // counters
        ++frameIndex;
    }

    std::cout << std::endl;
    std::cout << "End pose : " << pose << std::endl;
    std::cout << "Process terminated at frame " << frameIndex << std::endl;
    std::cout << std::endl;
    RGBD_Slam.show_statistics();

    cv::destroyAllWindows();
    exit(0);
}










