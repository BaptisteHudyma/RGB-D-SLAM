#ifndef RGB_SLAM_HPP
#define RGB_SLAM_HPP

#include "Pose.hpp"
#include "Parameters.hpp"
#include "LocalMap.hpp"
#include "Image_Features_Handler.hpp"
#include "PNPSolver.hpp"
#include "MotionModel.hpp"

#include <opencv2/core/core.hpp>
#include <deque>

namespace poseEstimation {

    class RGB_SLAM {
        public:
            RGB_SLAM(const Parameters &param);
            void reset();

            Pose track(const cv::Mat& imgRGB, const cv::Mat& imgDepth);

            enum eState
            {
                eState_NOT_INITIALIZED = 1,
                eState_TRACKING,
                eState_LOST
            };
            eState get_state() const { return this->state; }

        protected:
            bool need_new_triangulation();
            bool triangulation_policy_decreasing_matches();
            bool triangulation_policy_always_triangulate();
            bool triangulation_policy_map_size();

        private:
            Pose perform_tracking(const Pose& estimatedPose, Image_Features_Struct& features, bool& isTracking);

            Local_Map localMap;
            PNP_Solver pnpSolver;
            Image_Features_Handler featureHandler;
            Motion_Model motionModel;

            Parameters params;
            Pose lastPose;
            unsigned int frameNumber;
            std::deque<int> lastMatches;
            eState state;
            bool (RGB_SLAM::*triangulationPolicy)();

        private:
            //prevent backend copy
            RGB_SLAM(const RGB_SLAM&) = delete;
            RGB_SLAM& operator=(const RGB_SLAM&) = delete;
            enum
            {
                N_MATCHES_WINDOWS = 3
            };
    };
};

#endif
