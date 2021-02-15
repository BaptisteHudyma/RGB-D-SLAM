#ifndef THIRDP_GEODESIC_OPERATIONS
#define THIRDP_GEODESIC_OPERATIONS 

#include <opencv2/opencv.hpp>

namespace cv {

    // Fill the holes
    int fillHole(const cv::Mat& src, cv::Mat& dst);

    //Description: Morphological geodesic corrosion and corrosion reconstruction operations
    //Parameter:
    //masker input image, mark image
    //mask mask image
    //dst output image
    //se structure element
    //iterations The number of geodesic corrosion, when the default is-1, it is the corrosion reconstruction operation
    int GeodesicErosion(const InputArray masker, const InputArray mask, OutputArray& dst, InputArray se, int iterations = -1);

    //Description: morphological geodesic expansion and expansion reconstruction operations
    //Parameter:
    //masker input image, mark image
    //mask mask image
    //dst output image
    //se structure element
    //iterations The number of geodesic expansion, when the default is -1, it is the expansion reconstruction operation
    int GeodesicDilation(const InputArray masker,const InputArray mask, OutputArray& dst, InputArray se, int iterations = -1);
}

#endif
