//circle imshow
#include <opencv2/opencv.hpp>

#include "Pose_Optimisation.hpp"
#include "parameters.hpp"

#include "RGBD_SLAM.hpp"

// cerr cout
#include <iostream>

namespace rgbd_slam {

#define CLASS_ERR "<RGBD_SLAM> "

    RGBD_SLAM::RGBD_SLAM(const std::stringstream& dataPath, unsigned int imageWidth, unsigned int imageHeight) :
        _width(imageWidth),
        _height(imageHeight)
    {
        // Load parameters (once)
        Parameters::parse_file("");
        // Get intrinsics parameters
        std::stringstream calibPath, calibYAMLPath;
        calibPath << dataPath.str() << "calib_params.xml";
        calibYAMLPath << dataPath.str() << "calib_params.yaml";
        std::string finalPath = calibPath.str();

        // primitive connected graph creator
        _depthOps = new primitiveDetection::Depth_Operations(
                finalPath, 
                _width, 
                _height, 
                Parameters::get_depth_map_patch_size()
                );
        if (_depthOps == nullptr or not _depthOps->is_ok()) {
            std::cerr << CLASS_ERR << "Cannot load parameter files, exiting" << std::endl;
            exit(-1);
        }

        // init motion model
        _motionModel.reset();

        //local map
        _localMap = new map_management::Local_Map();

        //plane/cylinder finder
        _primitiveDetector = new primitiveDetection::Primitive_Detection(
                _height,
                _width,
                Parameters::get_depth_map_patch_size(),
                Parameters::get_maximum_plane_match_angle(),
                Parameters::get_maximum_merge_distance(),
                true
                );

        // Line segment detector
        //Should refine, scale, Gaussian filter sigma
        _lineDetector = new cv::LSD(cv::LSD_REFINE_NONE, 0.3, 0.9);

        // Point detector and matcher
        _pointMatcher = new utils::Key_Point_Extraction(Parameters::get_minimum_hessian());

        // kernel for various operations
        _kernel = cv::Mat::ones(3, 3, CV_8U);

        // set display colors
        set_color_vector();

        // set vars
        _meanMatTreatmentTime = 0.0;
        _meanTreatmentTime = 0.0;
        _meanLineTreatment = 0.0;
        _meanPoseTreatmentTime = 0.0;
        _totalFrameTreated = 0;

        if (_primitiveDetector == nullptr) {
            std::cerr << CLASS_ERR << " Instanciation of Primitive_Detector failed" << std::endl;
            exit(-1);
        }
        if (_lineDetector == nullptr) {
            std::cerr << CLASS_ERR << " Instanciation of Line_Detector failed" << std::endl;
            exit(-1);
        }

    }


    poseEstimation::Pose RGBD_SLAM::track(const cv::Mat& inputRgbImage, const cv::Mat& inputDepthImage, bool detectLines) 
    {
        cv::Mat depthImage = inputDepthImage.clone();
        cv::Mat rgbImage = inputRgbImage.clone();

        cv::Mat grayImage;
        cv::cvtColor(rgbImage, grayImage, cv::COLOR_BGR2GRAY);

        primitive_container primitives;


        //clean warp artefacts
        //cv::Mat newMat;
        //cv::morphologyEx(depthImage, newMat, cv::MORPH_CLOSE, kernel);
        //cv::medianBlur(newMat, newMat, 3);
        //cv::bilateralFilter(newMat, depthImage,  7, 31, 15);

        //project depth image in an organized cloud
        double t1 = cv::getTickCount();
        // organized 3D depth image
        Eigen::MatrixXf cloudArrayOrganized(_width * _height, 3);
        _depthOps->get_organized_cloud_array(depthImage, cloudArrayOrganized);
        double time_elapsed = (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
        _meanMatTreatmentTime += time_elapsed;

        // Run primitive detection 
        t1 = cv::getTickCount();
        _segmentationOutput = cv::Mat::zeros(depthImage.size(), uchar(0));    //primitive mask mat
        _primitiveDetector->find_primitives(cloudArrayOrganized, primitives, _segmentationOutput);
        time_elapsed = (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
        _meanTreatmentTime += time_elapsed;



        //associate primitives
        std::map<int, int> associatedIds;
        if(not _previousFramePrimitives.empty()) {
            //find matches between consecutive images
            //compare normals, superposed area (and colors ?)
            for(const std::unique_ptr<primitiveDetection::Primitive>& prim : primitives) {
                for(const std::unique_ptr<primitiveDetection::Primitive>& prevPrim : _previousFramePrimitives) {
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

        if(detectLines) { //detect lines in image
            cv::Mat outImage;
            compute_lines(grayImage, depthImage, outImage);
        }

        // exchange frames features
        _previousFramePrimitives.swap(primitives);
        _previousAssociatedIds.swap(associatedIds);

        // this frame points and  assoc
        t1 = cv::getTickCount();
        poseEstimation::Pose pose = this->compute_new_pose(rgbImage, depthImage);
        _meanPoseTreatmentTime += (cv::getTickCount() - t1) / (double)cv::getTickFrequency();

        _totalFrameTreated += 1;
        return pose;
    }

    void RGBD_SLAM::get_debug_image(const poseEstimation::Pose& camPose, const cv::Mat originalRGB, cv::Mat& debugImage, double elapsedTime, bool showPrimitiveMasks) {
        debugImage = originalRGB.clone();
        if (showPrimitiveMasks)
            _primitiveDetector->apply_masks(originalRGB, _colorCodes, _segmentationOutput, _previousFramePrimitives, debugImage, _previousAssociatedIds, elapsedTime);

        _localMap->get_debug_image(camPose, debugImage); 
    }


    const poseEstimation::Pose RGBD_SLAM::compute_new_pose (const cv::Mat& grayImage, const cv::Mat& depthImage) 
    {
        //get and refine pose
        poseEstimation::Pose refinedPose = _motionModel.predict_next_pose(_currentPose);
        //poseEstimation::Pose refinedPose(_currentPose);

        const utils::Keypoint_Handler& keypointObject = _pointMatcher->detect_keypoints(grayImage, depthImage);
        match_point_container matchedPoints = _localMap->find_matches(keypointObject);

        if (matchedPoints.size() > Parameters::get_minimum_point_count_for_optimization()) {
            double t1 = cv::getTickCount();

            // Vector to optimize: (0, 1, 2) is delta position, (3, 4, 5, 6) is rotation as a quaternion, 
            Eigen::VectorXd input(7);
            input[0] = 0;
            input[1] = 0;
            input[2] = 0;
            input[3] = 1;
            input[4] = 0;
            input[5] = 0;
            input[6] = 0;

            poseOptimisation::Pose_Functor pose_optimisation_functor(
                    poseOptimisation::Pose_Estimator(
                        input.size(), 
                        matchedPoints, 
                        refinedPose.get_position(),
                        refinedPose.get_orientation_quaternion()
                        )
                    );
            Eigen::LevenbergMarquardt<poseOptimisation::Pose_Functor, double> poseOptimisator( pose_optimisation_functor );

            // xtol     : tolerance for the norm of the solution vector
            // ftol     : tolerance for the norm of the vector function
            // gtol     : tolerance for the norm of the gradient of the error function
            // factor   : step bound for the diagonal shift
            // epsfcn   : error precision
            // maxfev   : maximum number of function evaluation
            poseOptimisator.parameters.maxfev = Parameters::get_maximum_optimization_iterations();

            const Eigen::LevenbergMarquardtSpace::Status endStatus = poseOptimisator.minimize(input);

            const quaternion endRotation(input[3], input[4], input[5], input[6]);
            const vector3 endTranslation(input[0], input[1], input[2]);

            refinedPose.update(endTranslation, endRotation);
            //refinedPose = poseEstimation::Pose(endTranslation, endRotation);

            if (endStatus == Eigen::LevenbergMarquardtSpace::Status::TooManyFunctionEvaluation)
            {
                const std::string message = poseOptimisation::get_human_readable_end_message(endStatus);
                std::cerr << matchedPoints.size() << " pts " << endTranslation.transpose() << " in " << poseOptimisator.iter << " iters. Result " << endStatus << " (" << message << ") in " << (cv::getTickCount() - t1) / cv::getTickFrequency() << std::endl;
            }

            _meanPoseOptimisationIterations += poseOptimisator.iter;
        }
        else
            std::cerr << "Not enough points for pose estimation: " << matchedPoints.size() << std::endl;

        //update motion model with refined pose
        _motionModel.update_model(refinedPose);

        _currentPose = refinedPose;

        _localMap->update(refinedPose, keypointObject);

        return refinedPose;
    }





    void RGBD_SLAM::set_color_vector() 
    {
        for(int i = 0; i < 100; i++) {
            cv::Vec3b color;
            color[0] = rand() % 255;
            color[1] = rand() % 255;
            color[2] = rand() % 255;
            _colorCodes.push_back(color);
        }

        // Add specific colors for planes
        _colorCodes[0][0] = 0; _colorCodes[0][1] = 0; _colorCodes[0][2] = 255;
        _colorCodes[1][0] = 255; _colorCodes[1][1] = 0; _colorCodes[1][2] = 204;
        _colorCodes[2][0] = 255; _colorCodes[2][1] = 100; _colorCodes[2][2] = 0;
        _colorCodes[3][0] = 0; _colorCodes[3][1] = 153; _colorCodes[3][2] = 255;
        // Add specific colors for cylinders
        _colorCodes[50][0] = 178; _colorCodes[50][1] = 255; _colorCodes[50][2] = 0;
        _colorCodes[51][0] = 255; _colorCodes[51][1] = 0; _colorCodes[51][2] = 51;
        _colorCodes[52][0] = 0; _colorCodes[52][1] = 255; _colorCodes[52][2] = 51;
        _colorCodes[53][0] = 153; _colorCodes[53][1] = 0; _colorCodes[53][2] = 255;
    }


    double get_percent_of_elapsed_time(double treatmentTime, double totalTimeElapsed) 
    {
        if (totalTimeElapsed <= 0)
            return 0;
        return std::round(treatmentTime / totalTimeElapsed * 10000) / 100;
    }


    void RGBD_SLAM::show_statistics(double meanFrameTreatmentTime) 
    {
        double pointCloudTreatmentTime = _meanMatTreatmentTime / _totalFrameTreated;
        std::cout << "Mean image to point cloud treatment time is " << pointCloudTreatmentTime << " seconds (" << get_percent_of_elapsed_time(pointCloudTreatmentTime, meanFrameTreatmentTime) << "%)" << std::endl;
        double planeTreatmentTime = _meanTreatmentTime / _totalFrameTreated;
        std::cout << "Mean primitive treatment time is " << planeTreatmentTime << " seconds (" << get_percent_of_elapsed_time(planeTreatmentTime, meanFrameTreatmentTime) << "%)" << std::endl;
        std::cout << std::endl;

        double lineDetectionTime = _meanLineTreatment / _totalFrameTreated;
        std::cout << "Mean line detection time is " << lineDetectionTime << " seconds (" << get_percent_of_elapsed_time(lineDetectionTime, meanFrameTreatmentTime) << "%)" << std::endl;
        double poseTreatmentTime = _meanPoseTreatmentTime / _totalFrameTreated;
        std::cout << "Mean pose estimation time is " << poseTreatmentTime << " seconds (" << get_percent_of_elapsed_time(poseTreatmentTime, meanFrameTreatmentTime) << "%)" << std::endl;

        std::cout << "Mean iteration for pose optimization: " << _meanPoseOptimisationIterations / _totalFrameTreated << std::endl;

        _pointMatcher->show_statistics(meanFrameTreatmentTime, _totalFrameTreated);
    }


    void RGBD_SLAM::compute_lines(const cv::Mat& grayImage, const cv::Mat& depthImage, cv::Mat& outImage)
    {
        double t1 = cv::getTickCount();

        //get lines
        line_vector lines;
        cv::Mat mask = depthImage > 0;

        _lineDetector->detect(grayImage, lines);

        //fill holes
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, _kernel);

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
                    cv::line(outImage, pt1, pt2, cv::Scalar(0, 0, 255), 1);
                else    //no depth data
                    cv::line(outImage, pt1, pt2, cv::Scalar(255, 0, 255), 1);
            }
            else
                //line with associated depth
                cv::line(outImage, pt1, pt2, cv::Scalar(0, 255, 255), 1);

        }
        _meanLineTreatment += (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
    }


} /* namespace poseEstimation */
