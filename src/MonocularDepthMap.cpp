
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "MonocularDepthMap.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

Monocular_Depth_Map::Monocular_Depth_Map(cv::Mat& firstImage) {
    //feature finder init
    int minHessian = 400;
    featureDetector = SURF::create( minHessian );
    featuresMatcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

    const int blocSize = 11;    //3 - 11, odd
    stereoDepthmapCompute = StereoSGBM::create();
    stereoDepthmapCompute->setMinDisparity(-16);
    stereoDepthmapCompute->setNumDisparities(16 * 2);   //> 0, %16
    stereoDepthmapCompute->setBlockSize(blocSize);
    stereoDepthmapCompute->setUniquenessRatio(5);      //5 - 15
    stereoDepthmapCompute->setDisp12MaxDiff(0);
    stereoDepthmapCompute->setPreFilterCap(32);
    stereoDepthmapCompute->setSpeckleWindowSize(50);   //50 - 200
    stereoDepthmapCompute->setSpeckleRange(2);          //1 - 2
    stereoDepthmapCompute->setP1(8 * blocSize * blocSize);
    stereoDepthmapCompute->setP2(32 * blocSize * blocSize);
    stereoDepthmapCompute->setMode(StereoSGBM::MODE_SGBM);


    stereoFilter = cv::ximgproc::createDisparityWLSFilter(stereoDepthmapCompute);
    stereoFilter->setLambda(8000);
    stereoFilter->setSigmaColor(1.8);    //0.8-2.0

    lastUndistorededImage = firstImage.clone();
    if(lastUndistorededImage.channels() > 1)
        cv::cvtColor(lastUndistorededImage, lastUndistorededImage, cv::COLOR_BGR2GRAY);


    //get first image descriptors
    indexOfLast = 0;
    featureDetector->detectAndCompute(lastUndistorededImage, cv::noArray(), keypoints[indexOfLast], descriptors[indexOfLast]); 

}



void Monocular_Depth_Map::get_monocular_depth(cv::Mat& image, cv::Mat& depthMap) {
    int thisIndex = 1 - indexOfLast;    //inverse index : 1 - 0 = 1; 0 - 1 = 1
    const float ratio_thresh = 0.9f;

    if(image.channels() > 1)
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    featureDetector->detectAndCompute(image, cv:: noArray(), keypoints[thisIndex], descriptors[thisIndex]); 

    std::vector< std::vector<DMatch> > knnMatches;
    featuresMatcher->knnMatch(descriptors[indexOfLast], descriptors[thisIndex], knnMatches, 2);
    //sort result and keep best N %
    //std::sort(knnMatches.begin(), knnMatches.end());
    //const int percToKeep = knnMatches.size() * 0.8;
    //knnMatches.erase(knnMatches.begin() + percToKeep, knnMatches.end());


    //-- Filter matches using the Lowe's ratio test
    std::vector<DMatch> goodMatches;
    std::vector<cv::Point2f> imagePoints1;
    std::vector<cv::Point2f> imagePoints2;
    for (size_t i = 0; i < knnMatches.size(); i++)
    {
        const std::vector<DMatch>& match = knnMatches[i];
        //closest point closest by multiplicator than second closest
        if (match[0].distance < ratio_thresh * match[1].distance)
        {
            int trainIdx = match[0].trainIdx;
            int queryIdx = match[0].queryIdx;

            goodMatches.push_back(match[0]);
            imagePoints2.push_back(keypoints[thisIndex][trainIdx].pt);
            imagePoints1.push_back(keypoints[indexOfLast][queryIdx].pt);
        }
    }
    //cv::drawMatches( lastUndistorededImage, keypoints[indexOfLast], image, keypoints[thisIndex], goodMatches, depthMap, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //get inliers
    cv::Mat fundMatrix = cv::findHomography(imagePoints1, imagePoints2, cv::FM_RANSAC);
    warpPerspective(lastUndistorededImage, lastUndistorededImage, fundMatrix, lastUndistorededImage.size());

    //compute depth disparity map
    cv::Mat disparityMapLeft;
    stereoDepthmapCompute->compute(lastUndistorededImage, image, disparityMapLeft);
    cv::Mat disparityMapRight;
    stereoDepthmapCompute->compute(image, lastUndistorededImage, disparityMapRight);

    stereoFilter->filter(disparityMapLeft, lastUndistorededImage, depthMap, disparityMapRight);
    cv::normalize(depthMap, depthMap, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    //cv::Mat confidenceMap = stereoFilter->getConfidenceMap();
    //confidenceMap.convertTo(confidenceMap, CV_8UC1);
    //hconcat(confidenceMap, depthMap, depthMap);

    //inverse indexes
    keypoints[indexOfLast].clear();
    descriptors[indexOfLast].release();
    indexOfLast = thisIndex;
    lastUndistorededImage = image.clone();
}





