#include "GeodesicOperations.hpp"

using namespace cv;

// Fill the holes
int cv::fillHole(const cv::Mat& src, cv::Mat& dst)
{
    cv::Mat m = cv::Mat(src.rows, src.cols, CV_8UC1);       // get F
 
         // first row 
    uchar *pxvec = m.ptr<uchar>(0);
    const uchar *pxvecSrc = src.ptr<uchar>(0);
    for (int i = 0; i < m.cols; i++)
        pxvec[i] = 255 - pxvecSrc[i];
 
         // the last line 
    uchar* pxvec2 = m.ptr<uchar>(m.rows-1);
    const uchar* pxvecSrc2 = src.ptr<uchar>(src.rows-1);
    for (int i = 0; i < m.cols; i++)
        pxvec2[i] = 255 - pxvecSrc2[i];
    
         // two columns
    for (int i = 1; i < m.rows - 1; i++)
    {
        uchar* pxvec3 = m.ptr<uchar>(i);
        const uchar* pxvecSrc3 = src.ptr<uchar>(i);
        pxvec3[0] = 255 - pxvecSrc2[0];
        pxvec3[m.cols-1] = 255 - pxvecSrc2[m.cols - 1];
    }
 
    cv::Mat mask;
    cv::bitwise_not(src, mask);                     // mask ，Ic
 
    uchar matrix_3x3[3][3] = { {1,1,1},{1,1,1},{1,1,1}};
    cv::Mat kernel3(Size(3, 3), CV_8UC1, matrix_3x3);   // se
    cv::Mat masker = m;
    GeodesicDilation(masker, mask, dst, kernel3);
 
    cv::bitwise_not(dst, dst);
    return 0;
}

//Description: Morphological geodesic corrosion and corrosion reconstruction operations
 //Parameter:
 //masker input image, mark image
 //mask mask image
 //dst output image
 //se structure element
 //iterations The number of geodesic corrosion, when the default is-1, it is the corrosion reconstruction operation
int cv::GeodesicErosion(const InputArray masker, const InputArray mask, OutputArray& dst, InputArray se, int iterations)
{
    if(iterations < 0)
    {
        cv::max(masker, mask, dst);
        cv::erode(dst, dst, se);
        cv::max(dst, mask, dst);
        Mat temp1, temp2;
        masker.copyTo(temp1);
        masker.copyTo(temp2);
        do
        {
            dst.copyTo(temp1);
            cv::erode(dst, dst, se);
            cv::max(dst, mask, dst);
            cv::compare(temp1, dst, temp2, cv::CMP_NE);
        }
        while (cv::sum(temp2).val[0] != 0);
        temp1.release();
        temp2.release();
        return 0;
    }
    else if (iterations == 0)
    {
        masker.copyTo(dst);
        return 0;
    }
    else
    {
                 //Ordinary geodesic corrosion
        cv::max(masker, mask, dst);
        cv::erode(dst, dst, se);
        cv::max(dst, mask, dst);
        for (int i = 1; i < iterations; i++)
        {
            cv::erode(dst, dst, se);
            cv::max(dst, mask, dst);
        }
        return 0;
    }
    return -1;
}
 
 //Description: morphological geodesic expansion and expansion reconstruction operations
 //Parameter:
 //masker input image, mark image
 //mask mask image
 //dst output image
 //se structure element
 //iterations The number of geodesic expansion, when the default is -1, it is the expansion reconstruction operation
int cv::GeodesicDilation(const InputArray masker,const InputArray mask, OutputArray& dst, InputArray se, int iterations)
{
    if (iterations < 0)
    {
        cv::min(masker, mask, dst);
        cv::dilate(dst, dst, se);
        cv::min(dst, mask, dst);
        Mat temp1, temp2;
        masker.copyTo(temp1);
        masker.copyTo(temp2);
        do
        {
            dst.copyTo(temp1);
            cv::dilate(dst, dst, se);
            cv::min(dst, mask, dst);
            cv::imshow("i", dst);
            cv::compare(temp1, dst, temp2, cv::CMP_NE);
        } while (cv::sum(temp2).val[0] != 0);
        temp1.release();
        temp2.release();
        return 0;
    }
    else if (iterations == 0)
    {
        masker.copyTo(dst);
        return 0;
    }
    else
    {
                 //Ordinary geodesic expansion
        cv::min(masker, mask, dst);
        cv::dilate(dst, dst, se);
        cv::min(dst, mask, dst);
        for (int i = 1; i < iterations; i++)
        {
            cv::dilate(dst, dst, se);
            cv::min(dst, mask, dst);
        }
        return 0;
    }
    return -1;
}
