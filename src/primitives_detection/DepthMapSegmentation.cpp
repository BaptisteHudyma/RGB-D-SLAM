#include "DepthMapSegmentation.hpp"

#include "GeodesicOperations.hpp"

using namespace primitiveDetection;


void get_segmented_components(const cv::Mat& edgeMap, const cv::Mat& kernel, cv::Mat& labeledComponents, int minArea=300) {
    cv::Mat edgeMapCleaned = edgeMap.clone();
    cv::fillHole(edgeMapCleaned, edgeMapCleaned);
    cv::morphologyEx(edgeMapCleaned, edgeMapCleaned, cv::MORPH_OPEN, kernel);

    labeledComponents= cv::Mat::zeros(edgeMapCleaned.size(), CV_8UC1);

    cv::Mat connectedCompo, stats, centroids;
    int segCount = cv::connectedComponentsWithStats(edgeMapCleaned, connectedCompo, stats, centroids);

    std::map<int, bool> largeComponents;
    for(int label = 1; label < segCount; ++label) {
        if(stats.at<int>(label, cv::CC_STAT_AREA) >= minArea)
            largeComponents[label] = true;
        else
            largeComponents[label] = false;
    }

    for (int r = 0; r < labeledComponents.rows; ++r) {
        for (int c = 0; c < labeledComponents.cols; ++c) {
            int label = connectedCompo.at<int>(r, c);
            if(not largeComponents[label])
                continue;
            labeledComponents.at<uchar>(r, c) = label;
        }
    }
}



void primitiveDetection::get_normal_map(const cv::Mat& depthMap, cv::Mat& normalMap) {
    assert(depthMap.type() == CV_32FC1);

    normalMap = cv::Mat::zeros(depthMap.size(), CV_32FC3);
    for(int x = 0; x < depthMap.rows; ++x) {
        const float* rowThis = depthMap.ptr<float>(x);
        const float* rowLow = depthMap.ptr<float>(x - 1);
        const float* rowAft = depthMap.ptr<float>(x + 1);

        cv::Vec3f* outRow = normalMap.ptr<cv::Vec3f>(x);
        for(int y = 0; y < depthMap.cols; ++y) {
            if(rowLow[y] <= 0 or rowAft[y] <= 0 or rowThis[y - 1] <= 0 or rowThis[y + 1] <= 0)
                continue;
            float dzdx = (rowAft[y] - rowLow[y]) / 2.0;
            float dzdy = (rowThis[y + 1] - rowThis[y - 1]) / 2.0;

            float normalizer = 1.0 / sqrt( pow(-dzdx, 2.0) + pow(-dzdy, 2.0) + 1.0);
            outRow[y] = cv::Vec3f(-dzdx * normalizer, -dzdy * normalizer, normalizer);
        }
    }
    //smooth normal map
    cv::medianBlur(normalMap, normalMap, 3);
}

void primitiveDetection::get_edge_masks(const cv::Mat& depthMap, const cv::Mat& normalMap, cv::Mat& edgeMap) {
    assert(depthMap.type() == CV_32FC1);

    assert(normalMap.type() == CV_32FC3 and normalMap.size() == depthMap.size());

    edgeMap= cv::Mat::zeros(depthMap.size(), CV_8UC1);
    const int neighborsX[8] = {-1, -1, -1,  0,  0,  1,  1,  1};
    const int neighborsY[8] = {-1,  0,  1, -1,  1, -1,  0,  1};

    int distanceCheck = 3;
    for(int x = distanceCheck; x < depthMap.rows - distanceCheck; x++) {
        const float* rowDepth = depthMap.ptr<float>(x);
        const cv::Vec3f* rowNormal = normalMap.ptr<cv::Vec3f>(x);
        for(int y = distanceCheck; y < depthMap.cols - distanceCheck; y++) {
            float vU = rowDepth[y];
            if(vU <= 0)
                continue;

            const cv::Vec3f& centerNormal = rowNormal[y];

            float minConcavity = 1;
            double maxNorm = 0;
            for(int i = 0; i < 8; i++) {
                const float depthValue = depthMap.at<float>(x + neighborsX[i], y + neighborsY[i]);
                if(depthValue <= 0)
                    continue;
                const cv::Vec3f& normal = normalMap.at<cv::Vec3f>(x + neighborsX[i], y + neighborsY[i]);

                float sub = depthValue - vU;
                float phiOperator = 1;
                if(sub <= 0) {
                    phiOperator = 0;
                    for(int j = 1; j <= distanceCheck; j++)
                        phiOperator += centerNormal.dot(normalMap.at<cv::Vec3f>(x + neighborsX[i] * j, y + neighborsY[i] * j)) / (distanceCheck);
                }
                minConcavity = std::min(minConcavity, phiOperator);
                maxNorm = std::max(maxNorm, cv::norm(sub * normal) );
            }
            uchar thresConc = (minConcavity > 0.94);
            uchar thresDist = maxNorm < (0.0012 + 0.0019 * pow(vU/10 - 0.4, 2.0));
            edgeMap.at<uchar>(x, y) = (thresConc & thresDist) * 255;
        }
    }
}

void primitiveDetection::get_segmented_depth_map(const cv::Mat& depthMap, cv::Mat& finalSegmented, const cv::Mat& kernel, double reducePourcent) {
    assert(depthMap.data and kernel.data);

    cv::Mat smallDepth, normalMap, edgeMap; 
    cv::resize(depthMap, smallDepth, cv::Size(), reducePourcent, reducePourcent);

    get_normal_map(smallDepth, normalMap);
    get_edge_masks(smallDepth, normalMap, edgeMap);
    get_segmented_components(edgeMap, kernel, finalSegmented);

    /*
    cv::resize(normalMap, normalMap, depthMap.size());
    normalMap = (normalMap * 0.5 + 0.5) * 255;
    normalMap.convertTo(normalMap, CV_8UC3);
    */
}



void primitiveDetection::draw_segmented_labels(const cv::Mat& segmentedImage, const std::vector<cv::Vec3b>& colors, cv::Mat& outputImage) {
    assert(segmentedImage.data and segmentedImage.data != outputImage.data);

    outputImage = cv::Mat::zeros(segmentedImage.size(), CV_8UC3);
    for (int r = 0; r < segmentedImage.rows; ++r) {
        for (int c = 0; c < segmentedImage.cols; ++c) {
            uchar label = segmentedImage.at<uchar>(r, c);
            outputImage.at<cv::Vec3b>(r, c) = colors[label];
        }
    }
}
