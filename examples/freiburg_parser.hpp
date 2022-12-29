

#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

struct ImageWithTimeStamp
{
    std::string imagePath = "";
    double imageTimeStamp = 0.0;
    bool isValid = false;   // indicates if this contains any data
};

struct GroundTruth
{
    double timeStamp = 0.0;
    Eigen::Vector3d position;
    Eigen::Quaterniond rotation;
    bool isValid = false;
};

struct Data {
    ImageWithTimeStamp rgbImage;
    ImageWithTimeStamp depthImage;
    GroundTruth groundTruth;
};


class DatasetParser
{
    public:
    typedef std::map<double, std::string> dataMap;
    typedef std::map<double, GroundTruth> groundTruthMap;
    static std::vector<Data> parse_dataset(const std::string& rgbImageListPath, const std::string& depthImageListPath, const std::string& groundTruthListPath)
    {
        dataMap rgbFile = parse_file(rgbImageListPath);
        dataMap depthFile = parse_file(depthImageListPath);
        groundTruthMap groundTruth = parse_ground_truth(groundTruthListPath);

        return associate_data(rgbFile, depthFile, groundTruth);
    }

    protected:

    static dataMap parse_file(const std::string& filePath)
    {
        std::ifstream imagesFile(filePath);

        dataMap res;
        for(std::string line; std::getline(imagesFile, line); ) 
        {
            if (line[0] == '#')
                continue;   // comment in the file
            std::istringstream inputString(line);

            double timeStamp = 0.0;
            std::string imagePath = "";
            inputString >> timeStamp >> imagePath;

            if (imagePath != "" and timeStamp > 0)
            {
                res.emplace(timeStamp, imagePath);
            }
        }
        return res;
    }

    static groundTruthMap parse_ground_truth(const std::string& groundTruthListPath)
    {
        std::ifstream groundTruthFile(groundTruthListPath);

        groundTruthMap res;
        for(std::string line; std::getline(groundTruthFile, line); ) 
        {
            if (line[0] == '#')
                continue;   // comment in the file
            std::istringstream inputString(line);

            GroundTruth gt;
            inputString >> gt.timeStamp >>
            gt.position.x() >> gt.position.y() >> gt.position.z() >>
            gt.rotation.x() >> gt.rotation.y() >> gt.rotation.z() >> gt.rotation.w();

            res.emplace(gt.timeStamp, gt);
        }
        return res;
    }

    static std::vector<Data> associate_data(dataMap& rgbImagesData, dataMap& depthImagesData, groundTruthMap& groundTruthData, const double timeOffset = 0.0, const double maxDifference = 0.1)
    {
        struct TimeStampScore {
            double score;
            double rgbTimeStamp;
            double depthTimeStamp;
        };

        // compute timestamp distance score
        std::vector<TimeStampScore> timeStampAssociation;
        timeStampAssociation.reserve(rgbImagesData.size() * depthImagesData.size());
        for(const std::pair<double, std::string> rgbData : rgbImagesData)
        {
            const double rgbTimeStamp = rgbData.first;
            bool readyToBreak = false;  //save a lot of processing power when frames are organized
            for(const std::pair<double, std::string> depthData : depthImagesData)
            {
                const double depthTimeStamp = depthData.first;

                TimeStampScore t;
                t.score = std::abs(rgbTimeStamp - (depthTimeStamp + timeOffset));
                if (t.score < maxDifference)
                {
                    t.rgbTimeStamp = rgbTimeStamp;
                    t.depthTimeStamp = depthTimeStamp;
                    timeStampAssociation.emplace_back(t);

                    readyToBreak = true;
                }
                else if(readyToBreak)
                    break;
            }
        }
        // sort by score
        std::sort(timeStampAssociation.begin(), timeStampAssociation.end(), [](const TimeStampScore& a, const TimeStampScore& b) {
            return a.score < b.score and a.rgbTimeStamp < b.rgbTimeStamp;
        });

        const size_t numberOfRGBData = rgbImagesData.size();
        const size_t numberOfDepthData = depthImagesData.size();
        std::vector<Data> finalSorted;
        finalSorted.reserve(rgbImagesData.size());
        for(const TimeStampScore& ts : timeStampAssociation)
        {
            const dataMap::const_iterator rgbIt = rgbImagesData.find(ts.rgbTimeStamp);
            const dataMap::const_iterator depthIt = depthImagesData.find(ts.depthTimeStamp);

            if(rgbIt != rgbImagesData.cend() and depthIt != depthImagesData.cend())
            {
                Data d;
                d.rgbImage.imagePath = rgbIt->second;
                d.rgbImage.imageTimeStamp = ts.rgbTimeStamp;
                d.rgbImage.isValid = true;

                d.depthImage.imagePath = depthIt->second;
                d.depthImage.imageTimeStamp = ts.depthTimeStamp;
                d.depthImage.isValid = true;

                finalSorted.emplace_back(d);
                rgbImagesData.erase(rgbIt);
                depthImagesData.erase(depthIt);
            }
            // RGBD without depth
            /*else if(rgbIt != rgbImagesData.cend())
            {
                Data d;
                d.rgbImage.imagePath = rgbIt->second;
                d.rgbImage.imageTimeStamp = ts.rgbTimeStamp;
                d.rgbImage.isValid = true;

                d.depthImage.isValid = false;

                finalSorted.emplace_back(d);
                rgbImagesData.erase(rgbIt);
            }
            // depth without rgb (ignore)
            else if(depthIt != depthImagesData.cend())
            {
                Data d;
                d.depthImage.imagePath = depthIt->second;
                d.depthImage.imageTimeStamp = ts.depthTimeStamp;
                d.depthImage.isValid = true;

                d.rgbImage.isValid = false;

                finalSorted.emplace_back(d);
                depthImagesData.erase(depthIt);
            }*/
        }

        std::sort(finalSorted.begin(), finalSorted.end(), [](const Data& a, const Data& b) {
            return a.rgbImage.imageTimeStamp < b.rgbImage.imageTimeStamp;
        });

        if (rgbImagesData.size() > 0)
            std::cout << "Used " << (numberOfRGBData - rgbImagesData.size()) << " over " << numberOfRGBData << " RGB images" << std::endl;
        if (depthImagesData.size() > 0)
            std::cout << "Used " << (numberOfDepthData - depthImagesData.size())  << " over " << numberOfDepthData << " depth images" << std::endl;

        const size_t groundTruthCount = groundTruthData.size();
        // associate timestamps
        for(Data& data : finalSorted)
        {
            const double rgbTimeStamp = data.rgbImage.imageTimeStamp;
            GroundTruth closestGT;
            double closestScore = std::numeric_limits<double>::max();
            bool readyToBreak = false;  //save a lot of processing power when frames are organized
            for(const std::pair<double, GroundTruth> groundTruth : groundTruthData)
            {
                const double gtTimeStamp = groundTruth.first;

                const double score = std::abs(rgbTimeStamp - (gtTimeStamp + timeOffset));

                if(score < maxDifference)
                {
                    readyToBreak = true;
                    if (score < closestScore)
                    {
                        closestScore = score;
                        closestGT.position = groundTruth.second.position;
                        closestGT.rotation = groundTruth.second.rotation;
                        closestGT.timeStamp = groundTruth.second.timeStamp;
                    }
                }
                else if(readyToBreak)
                    break;
            }

            if (closestScore < std::numeric_limits<double>::max())
            {
                data.groundTruth.isValid = true;
                data.groundTruth.position = closestGT.position;
                data.groundTruth.rotation = closestGT.rotation;
                data.groundTruth.timeStamp = closestGT.timeStamp;

                groundTruthData.erase(closestGT.timeStamp);
            }
            else {
                data.groundTruth.isValid = false;
            }
        }

        if (groundTruthData.size() > 0)
            std::cout << "Uses " << (groundTruthCount - groundTruthData.size()) << " over " << groundTruthCount << " ground truth" << std::endl;

        return finalSorted;
    }

};
