

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

struct Data {
    ImageWithTimeStamp rgbImage;
    ImageWithTimeStamp depthImage;
};


class DatasetParser
{
    public:
    typedef std::map<double, std::string> dataMap;
    static std::vector<Data> parse_dataset(const std::string& rgbImageListPath, const std::string& depthImageListPath)
    {
        dataMap rgbFile = parse_file(rgbImageListPath);
        dataMap depthFile = parse_file(depthImageListPath);

        return associate_data(rgbFile, depthFile);
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

    static std::vector<Data> associate_data(dataMap& rgbImagesData, dataMap& depthImagesData, const double timeOffset = 0.0, const double maxDifference = 0.02)
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
                }
            }
        }
        // sort by score
        std::sort(timeStampAssociation.begin(), timeStampAssociation.end(), [](const TimeStampScore& a, const TimeStampScore& b) {
            return a.score < b.score;
        });


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
            else if(rgbIt != rgbImagesData.cend())
            {
                Data d;
                d.rgbImage.imagePath = rgbIt->second;
                d.rgbImage.imageTimeStamp = ts.rgbTimeStamp;
                d.rgbImage.isValid = true;

                d.depthImage.isValid = false;

                finalSorted.emplace_back(d);
                rgbImagesData.erase(rgbIt);
            }
            //else: depth without rgb (ignore)
        }

        std::sort(finalSorted.begin(), finalSorted.end(), [](const Data& a, const Data& b) {
            return a.rgbImage.imageTimeStamp < b.rgbImage.imageTimeStamp;
        });

        return finalSorted;
    }

};
