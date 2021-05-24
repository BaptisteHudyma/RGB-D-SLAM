#ifndef LOCAL_MAP_HPP
#define LOCAL_MAP_HPP

#include "Pose.hpp"
#include "Parameters.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

namespace poseEstimation {
    class Image_Features_Struct;
    class Image_Features_Handler;

    class Local_Map {

        public:
            Local_Map(const Parameters& voparams,
                    Image_Features_Handler* featureHandler);
            ~Local_Map();
            void reset();

            void update_with_new_triangulation(const Pose& camPose,
                    Image_Features_Struct& features, bool dontStage = false);
            

            void clean_untracked_points(Image_Features_Struct& features);
            void update_staged_map_points(const Pose& camPose, Image_Features_Struct& features);


        public: //getters
            unsigned int get_map_size() const { return _mapPoints.size(); }
            unsigned int get_staged_points_count() const { return _stagedPoints.size(); }
            int find_matches(const Pose &camPose, Image_Features_Struct& left_struct,
                     vector3_array& out_map_points, std::vector<int>& out_matches_left);

        private:
            struct mapPoint
            {
                EIGEN_MAKE_ALIGNED_OPERATOR_NEW
                cv::Mat descriptor;   // descriptor for the keypoint used initially to triangulate this map point
                vector3 position; // world position of this point
                int counter;          // depending on context, if the point is staged then this is the number of frames it has been tracked while staged. If it is a map point, then this is the number of times it failed tracking.
                int age;              // number of frames this map point was successfully tracked and thus used in pose estimation
                int match_idx;
            };
            typedef std::vector<mapPoint, Eigen::aligned_allocator<mapPoint>> mapPointArray;


            Parameters _voParams;
            Image_Features_Handler *_featuresHandler;

            mapPointArray _mapPoints;
            mapPointArray _stagedPoints;


        protected:

            //void triangulate(const Pose &camPose, Image_Features_Struct& features, Image_Features_Struct& featuresRight, mapPointArray& outPoints);

            void triangulate_rgbd(const Pose& camPose, Image_Features_Struct& features, mapPointArray& outPoints);

    };

}

#endif
