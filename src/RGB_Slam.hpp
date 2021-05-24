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
            eState get_state() const { return _state; }

        protected:
            bool need_new_triangulation();
            bool triangulation_policy_decreasing_matches();
            bool triangulation_policy_always_triangulate();
            bool triangulation_policy_map_size();

        private:
            Pose perform_tracking(const Pose& estimatedPose, Image_Features_Struct& features, bool& isTracking);

            Parameters _params;
            Image_Features_Handler _featureHandler;
            Local_Map _localMap;
            PNP_Solver _pnpSolver;

            Pose _lastPose;
            unsigned int _frameNumber;
            std::deque<long unsigned int> _lastMatches;
            eState _state;

            bool (RGB_SLAM::*triangulationPolicy)();

            Motion_Model _motionModel;


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
