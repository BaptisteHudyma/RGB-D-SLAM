//circle imshow
#include <opencv2/opencv.hpp>

#include "Pose_Optimisation.hpp"

#include "RGBD_SLAM.hpp"

#include <iostream>

namespace primitiveDetection {

#define CLASS_ERR "<RGBD_SLAM> "

    RGBD_SLAM::RGBD_SLAM(const std::stringstream& dataPath, unsigned int imageWidth, unsigned int imageHeight, unsigned int minHessian, double maxMatchDistance) :
        _width(imageWidth),
        _height(imageHeight),
        _maxMatchDistance(maxMatchDistance)
    {
        if (_maxMatchDistance <= 0 or _maxMatchDistance > 1) {
            std::cerr << CLASS_ERR << "Maximum matching distance disparity must be between 0 and 1" << std::endl;
            exit(-1);
        }


        // Get intrinsics parameters
        std::stringstream calibPath, calibYAMLPath;
        calibPath << dataPath.str() << "calib_params.xml";
        calibYAMLPath << dataPath.str() << "calib_params.yaml";
        std::string finalPath = calibPath.str();

        // primitive connected graph creator
        _depthOps = new Depth_Operations(finalPath, _width, _height, PATCH_SIZE);
        if (_depthOps == nullptr or not _depthOps->is_ok()) {
            std::cerr << CLASS_ERR << "Cannot load parameter files, exiting" << std::endl;
            exit(-1);
        }

        // init motion model
        _motionModel.reset();


        // Create feature extractor and matcher
        _featureDetector = cv::xfeatures2d::SURF::create( minHessian );
        _featuresMatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);


        //plane/cylinder finder
        _primitiveDetector = new Primitive_Detection(_height, _width, PATCH_SIZE, COS_ANGLE_MAX, MAX_MERGE_DIST, true);

        // Line segment detector
        //Should refine, scale, Gaussian filter sigma
        _lineDetector = new cv::LSD(cv::LSD_REFINE_NONE, 0.3, 0.9);

        // kernel for various operations
        _kernel = cv::Mat::ones(3, 3, CV_8U);

        // set display colors
        set_color_vector();

        // set vars
        _maxTreatTime = 0.0;
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
        _maxTreatTime = std::max(_maxTreatTime, time_elapsed);



        //associate primitives
        std::map<int, int> associatedIds;
        if(not _previousFramePrimitives.empty()) {
            //find matches between consecutive images
            //compare normals, superposed area (and colors ?)
            for(const std::unique_ptr<Primitive>& prim : primitives) {
                for(const std::unique_ptr<Primitive>& prevPrim : _previousFramePrimitives) {
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
            t1 = cv::getTickCount();

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
                        cv::line(rgbImage, pt1, pt2, cv::Scalar(0, 0, 255), 1);
                    else    //no depth data
                        cv::line(rgbImage, pt1, pt2, cv::Scalar(255, 0, 255), 1);
                }
                else
                    //line with associated depth
                    cv::line(rgbImage, pt1, pt2, cv::Scalar(0, 255, 255), 1);

            }
            _meanLineTreatment += (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
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

    void RGBD_SLAM::get_debug_image(const cv::Mat originalRGB, cv::Mat& debugImage, double elapsedTime, bool showPrimitiveMasks) {
        debugImage = originalRGB.clone();
        if (showPrimitiveMasks)
            _primitiveDetector->apply_masks(originalRGB, _colorCodes, _segmentationOutput, _previousFramePrimitives, debugImage, _previousAssociatedIds, elapsedTime);

        if (_lastFrameKeypoints.size() > 0) {
            for (const std::pair<unsigned int, poseEstimation::vector3> pair : _lastFrameKeypoints) {
                const poseEstimation::vector3 point = pair.second;
                if (point.z() <= 0) {
                    cv::circle(debugImage, cv::Point(point.x(), point.y()), 4, cv::Scalar(0, 255, 255), 1);
                }
                else {
                    // no depth
                    cv::circle(debugImage, cv::Point(point.x(), point.y()), 4, cv::Scalar(255, 255, 0), 1);
                }
            }
        }

    }


    void RGBD_SLAM::get_cleaned_keypoint(const cv::Mat& depthImage, const std::vector<cv::KeyPoint>& kp, keypoint_container& cleanedPoints) 
    {
        cv::Rect lastRect(cv::Point(), depthImage.size());
        unsigned int i = 0;
        for (const cv::KeyPoint& keypoint : kp) {
            const cv::Point2f& pt = keypoint.pt;
            if (lastRect.contains(pt)) {
                const float depth = depthImage.at<float>(pt.y, pt.x);
                if (depth > 0) {
                    cleanedPoints.emplace( i, poseEstimation::vector3(pt.x, pt.y, depth) );
                }
                else {
                    cleanedPoints.emplace( i, poseEstimation::vector3(pt.x, pt.y, 0) );
                }
            }
            ++i;
        }
    }


    const poseEstimation::Pose RGBD_SLAM::compute_new_pose (const cv::Mat& grayImage, const cv::Mat& depthImage) 
    {
        std::vector<cv::KeyPoint> frameKeypoints;
        cv::Mat frameDescriptors;

        _featureDetector->detectAndCompute(grayImage, cv:: noArray(), frameKeypoints, frameDescriptors); 

        keypoint_container cleanedKp;
        get_cleaned_keypoint(depthImage, frameKeypoints, cleanedKp);

        if (_lastFrameKeypoints.size() <= 0) {
            //first call, or tracking lost
            _lastFrameKeypoints.swap(cleanedKp);
            _lastFrameDescriptors = frameDescriptors;
            return _currentPose;
        }
        else if (frameKeypoints.size() <= 3) {
            std::cout << "Not enough features detected for knn matching" << std::endl;
            return _currentPose;
        }

        //get and refine pose
        poseEstimation::Pose refinedPose = _motionModel.predict_next_pose(_currentPose);

        const poseEstimation::matched_point_container matchedPoints = this->get_good_matches(cleanedKp, frameDescriptors);

        if (matchedPoints.size() > 5) {
            double t1 = cv::getTickCount();

            Eigen::VectorXd input(7);
            input[0] = 0;
            input[1] = 0;
            input[2] = 0;
            input[3] = 1;
            input[4] = 0;
            input[5] = 0;
            input[6] = 0;

            poseEstimation::Pose_Functor pf(poseEstimation::Pose_Estimator(input.size(), matchedPoints));
            Eigen::LevenbergMarquardt<poseEstimation::Pose_Functor, double> lm( pf );
            lm.parameters.epsfcn = 1e-5;
            lm.parameters.maxfev= 1024;

            Eigen::LevenbergMarquardtSpace::Status endStatus = lm.minimize(input);
            const std::string message = get_human_readable_end_message(endStatus);

            poseEstimation::quaternion endRotation(input[3], input[4], input[5], input[6]);
            endRotation.normalize();
            poseEstimation::vector3 endTranslation(input[0], input[1], input[2]);

            refinedPose.update(endTranslation, endRotation);

            std::cout << matchedPoints.size() << " pts " << endTranslation.transpose() << " in " << lm.iter << " iters. Result " << endStatus << " (" << message << ") in " << (cv::getTickCount() - t1) / cv::getTickFrequency() << std::endl;

        }
        else
            std::cerr << "Not enough points for pose estimation: " << matchedPoints.size() << std::endl;



        //update motion model with refined pose
        _motionModel.update_model(refinedPose);


        _lastFrameKeypoints.swap(cleanedKp);
        _lastFrameDescriptors = frameDescriptors;

        _currentPose = refinedPose;

        return refinedPose;
    }

    const poseEstimation::matched_point_container RGBD_SLAM::get_good_matches(keypoint_container& thisFrameKeypoints, cv::Mat& thisFrameDescriptors)
    {
        std::vector< std::vector<cv::DMatch> > knnMatches;
        _featuresMatcher->knnMatch(_lastFrameDescriptors, thisFrameDescriptors, knnMatches, 2);

        poseEstimation::matched_point_container matchedPoints;
        for (size_t i = 0; i < knnMatches.size(); i++)
        {
            const std::vector<cv::DMatch>& match = knnMatches[i];
            //check if point is a good match by checking it's distance to the second best matched point
            if (match[0].distance < _maxMatchDistance * match[1].distance)
            {
                int trainIdx = match[0].trainIdx;   //last frame key point
                int queryIdx = match[0].queryIdx;   //this frame key point

                if (_lastFrameKeypoints.contains(trainIdx) and thisFrameKeypoints.contains(queryIdx)) {
                    poseEstimation::point_pair newMatch = std::make_pair(
                            _lastFrameKeypoints[trainIdx],
                            thisFrameKeypoints[queryIdx]
                            );
                    matchedPoints.push_back(newMatch);
                }
            }
        }
        return matchedPoints;
    }


    const std::string RGBD_SLAM::get_human_readable_end_message(Eigen::LevenbergMarquardtSpace::Status status) 
    {
        switch(status) {
            case Eigen::LevenbergMarquardtSpace::Status::NotStarted :
                return "not started";
            case Eigen::LevenbergMarquardtSpace::Status::Running :
                return "running";
            case Eigen::LevenbergMarquardtSpace::Status::ImproperInputParameters :
                return "improper input parameters";
            case Eigen::LevenbergMarquardtSpace::Status::RelativeReductionTooSmall :
                return "relative reduction too small";
            case Eigen::LevenbergMarquardtSpace::Status::RelativeErrorTooSmall :
                return "relative error too small";
            case Eigen::LevenbergMarquardtSpace::Status::RelativeErrorAndReductionTooSmall :
                return "relative error and reduction too small";
            case Eigen::LevenbergMarquardtSpace::Status::CosinusTooSmall :
                return "cosinus too small";
            case Eigen::LevenbergMarquardtSpace::Status::TooManyFunctionEvaluation :
                return "too many function evaluation";
            case Eigen::LevenbergMarquardtSpace::Status::FtolTooSmall :
                return "xtol too small";
            case Eigen::LevenbergMarquardtSpace::Status::XtolTooSmall :
                return "ftol too small";
            case Eigen::LevenbergMarquardtSpace::Status::GtolTooSmall :
                return "gtol too small";
            case Eigen::LevenbergMarquardtSpace::Status::UserAsked :
                return "user asked";
            default:
                return "error: empty message";
        }
        return std::string("");
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

    void RGBD_SLAM::show_statistics() 
    {
        std::cout << "Mean image to point cloud treatment time is " << _meanMatTreatmentTime / _totalFrameTreated << std::endl;
        std::cout << "Mean plane treatment time is " << _meanTreatmentTime / _totalFrameTreated << std::endl;
        std::cout << "max treat time is " << _maxTreatTime << std::endl;
        std::cout << std::endl;
        std::cout << "Mean line detection time is " << _meanLineTreatment / _totalFrameTreated << std::endl;
        std::cout << "Mean pose estimation time is " << _meanPoseTreatmentTime / _totalFrameTreated << std::endl;

    }

} /* namespace poseEstimation */
