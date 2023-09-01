

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

// Can return RGB frames without depth image
#define USE_RGB_WITHOUT_DEPTH
// Can return depth image without RGB
// #define USE_DEPTH_WITHOUT_RGB

struct ImageWithTimeStamp
{
    std::string imagePath;
    double imageTimeStamp = 0.0;
    bool isValid = false; // indicates if this contains any data
};

struct GroundTruth
{
    double timeStamp = 0.0;
    Eigen::Vector3d position;
    Eigen::Quaterniond rotation;
    bool isValid = false;
};

struct Data
{
    ImageWithTimeStamp rgbImage;
    ImageWithTimeStamp depthImage;
    GroundTruth groundTruth;
};

/**
 * \brief This class parses a given TUM dataset and return the associated depth and rgb images, along with the ground
 * truth if available
 */
class DatasetParser
{
  public:
    using dataMap = std::map<double, std::string>;
    using groundTruthMap = std::map<double, GroundTruth>;

    static std::vector<Data> parse_dataset(const std::string& rgbImageListPath,
                                           const std::string& depthImageListPath,
                                           const std::string& groundTruthListPath)
    {
        dataMap rgbFile = parse_file(rgbImageListPath);
        dataMap depthFile = parse_file(depthImageListPath);
        groundTruthMap groundTruth = parse_ground_truth(groundTruthListPath);

        return associate_data(rgbFile, depthFile, groundTruth);
    }

    static std::vector<Data> parse_association_file(const std::string& dataPath, const std::string& groundTruthListPath)
    {
        std::vector<Data> data;
        std::ifstream associationFile(dataPath + "associations.txt");
        if (not associationFile.is_open())
        {
            return data;
        }

        const groundTruthMap& groundTruth = DatasetParser::parse_ground_truth(groundTruthListPath);

        for (std::string line; std::getline(associationFile, line);)
        {
            if (line[0] == '#')
                continue; // comment in the file

            std::istringstream inputString(line);

            double depthTimestamp;
            double rgbTimestamp;
            std::string depthPath;
            std::string rgbPath;
            inputString >> depthTimestamp >> depthPath >> rgbTimestamp >> rgbPath;

            if (depthPath != "" and rgbPath != "")
            {
                Data newData;
                newData.depthImage.imagePath = depthPath;
                newData.depthImage.isValid = true;
                newData.depthImage.imageTimeStamp = depthTimestamp;

                newData.rgbImage.imagePath = rgbPath;
                newData.rgbImage.isValid = true;
                newData.rgbImage.imageTimeStamp = rgbTimestamp;

                const auto& groundTruthElement = groundTruth.find(rgbTimestamp);
                if (groundTruthElement != groundTruth.end())
                {
                    newData.groundTruth = groundTruth.at(rgbTimestamp);
                    newData.groundTruth.isValid = true;
                }

                data.emplace_back(newData);
            }
        }
        return data;
    }

  protected:
    static groundTruthMap parse_ground_truth(const std::string& groundTruthListPath)
    {
        std::ifstream groundTruthFile(groundTruthListPath);

        groundTruthMap res;
        for (std::string line; std::getline(groundTruthFile, line);)
        {
            if (line[0] == '#')
                continue; // comment in the file
            std::istringstream inputString(line);

            GroundTruth gt;
            inputString >> gt.timeStamp >> gt.position.x() >> gt.position.y() >> gt.position.z() >> gt.rotation.x() >>
                    gt.rotation.y() >> gt.rotation.z() >> gt.rotation.w();

            res.try_emplace(gt.timeStamp, gt);
        }
        return res;
    }

    static dataMap parse_file(const std::string& filePath)
    {
        std::ifstream imagesFile(filePath);

        dataMap res;
        for (std::string line; std::getline(imagesFile, line);)
        {
            if (line[0] == '#')
                continue; // comment in the file
            std::istringstream inputString(line);

            double timeStamp = 0.0;
            std::string imagePath;
            inputString >> timeStamp >> imagePath;

            if (imagePath != "" and timeStamp > 0)
            {
                res.try_emplace(timeStamp, imagePath);
            }
        }
        return res;
    }

    static std::vector<Data> associate_data(dataMap& rgbImagesData,
                                            dataMap& depthImagesData,
                                            groundTruthMap& groundTruthData,
                                            const double depthTimeOffset = 0.0,
                                            const double maxDifference = 0.15)
    {
        struct TimeStampScore
        {
            double score;
            double rgbTimeStamp;
            double depthTimeStamp;
        };

        // compute timestamp distance score
        std::vector<TimeStampScore> timeStampAssociation;
        timeStampAssociation.reserve(rgbImagesData.size() * depthImagesData.size());
        for (const std::pair<double, std::string> rgbData: rgbImagesData)
        {
            const double rgbTimeStamp = rgbData.first;
            bool readyToBreak = false; // save a lot of processing power when frames are organized
            for (const std::pair<double, std::string> depthData: depthImagesData)
            {
                const double depthTimeStamp = depthData.first;

                TimeStampScore t;
                t.score = std::abs(rgbTimeStamp - (depthTimeStamp + depthTimeOffset));
                if (t.score < maxDifference)
                {
                    t.rgbTimeStamp = rgbTimeStamp;
                    t.depthTimeStamp = depthTimeStamp;
                    timeStampAssociation.emplace_back(t);

                    readyToBreak = true;
                }
                else if (readyToBreak)
                    break;
            }
        }
        // sort by score
        std::sort(timeStampAssociation.begin(),
                  timeStampAssociation.end(),
                  [](const TimeStampScore& a, const TimeStampScore& b) {
                      return a.score < b.score;
                  });

        const size_t numberOfRGBData = rgbImagesData.size();
        const size_t numberOfDepthData = depthImagesData.size();
        std::vector<Data> finalSorted;
        finalSorted.reserve(rgbImagesData.size());
        for (const TimeStampScore& ts: timeStampAssociation)
        {
            const dataMap::const_iterator rgbIt = rgbImagesData.find(ts.rgbTimeStamp);
            const dataMap::const_iterator depthIt = depthImagesData.find(ts.depthTimeStamp);

            if (rgbIt != rgbImagesData.cend() and depthIt != depthImagesData.cend())
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
        }

#ifdef USE_RGB_WITHOUT_DEPTH
        // RGBD without depth
        for (dataMap::const_iterator it = rgbImagesData.cbegin(); it != rgbImagesData.cend();)
        {
            Data d;
            d.rgbImage.imagePath = it->second;
            d.rgbImage.imageTimeStamp = it->first;
            d.rgbImage.isValid = true;

            d.depthImage.isValid = false;

            finalSorted.emplace_back(d);
            rgbImagesData.erase(it++);
        }
#endif

#ifdef USE_DEPTH_WITHOUT_RGB
        // depth without rgb (ignore)
        for (dataMap::const_iterator it = depthImagesData.cbegin(); it != depthImagesData.cend();)
        {
            Data d;
            d.depthImage.imagePath = it->second;
            d.depthImage.imageTimeStamp = it->first;
            d.depthImage.isValid = true;

            d.rgbImage.isValid = false;

            finalSorted.emplace_back(d);
            depthImagesData.erase(it++);
        }
#endif

        std::sort(finalSorted.begin(), finalSorted.end(), [](const Data& a, const Data& b) {
            // rgb timestamp is smaller, OR depth timestamp is smaller
            return ((a.rgbImage.isValid and b.rgbImage.isValid) and
                    a.rgbImage.imageTimeStamp < b.rgbImage.imageTimeStamp) or
                   ((a.depthImage.isValid and b.depthImage.isValid) and
                    a.depthImage.imageTimeStamp < b.depthImage.imageTimeStamp);
            /*or   ((a.rgbImage.isValid and b.depthImage.isValid) and a.rgbImage.imageTimeStamp <
            b.depthImage.imageTimeStamp) or   ((a.depthImage.isValid and b.rgbImage.isValid)   and
            a.depthImage.imageTimeStamp < b.rgbImage.imageTimeStamp);*/
        });

        std::cout << "Used " << (numberOfRGBData - rgbImagesData.size()) << " over " << numberOfRGBData << " RGB images"
                  << std::endl;
        std::cout << "Used " << (numberOfDepthData - depthImagesData.size()) << " over " << numberOfDepthData
                  << " depth images" << std::endl;

        const size_t groundTruthCount = groundTruthData.size();
        // associate timestamps
        for (Data& data: finalSorted)
        {
            const double timeStamp = data.rgbImage.imageTimeStamp;

            GroundTruth closestGT;
            double closestScore = std::numeric_limits<double>::max();
            bool readyToBreak = false; // save a lot of processing power when frames are organized
            for (const std::pair<double, GroundTruth> groundTruth: groundTruthData)
            {
                const double gtTimeStamp = groundTruth.first;

                const double score = std::abs(timeStamp - gtTimeStamp);

                if (score < 0.05)
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
                else if (readyToBreak)
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
            else
            {
                data.groundTruth.isValid = false;
            }
        }

        std::cout << "Uses " << (groundTruthCount - groundTruthData.size()) << " over " << groundTruthCount
                  << " ground truth" << std::endl;

        return finalSorted;
    }
};
