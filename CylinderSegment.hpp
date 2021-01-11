#ifndef CYLINDER_SEGMENT_H
#define CYLINDER_SEGMENT_H

#include "PlaneSegment.hpp"
#include <vector>
#include <memory>
#include <Eigen/Dense>


namespace planeDetection {
    const float CYLINDER_RANSAC_SQR_MAX_DIST = 0.0225;    //square of 15%
    const float CYLINDER_SCORE_MIN = 100;
    typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>  MatrixXb;


    class Cylinder_Segment {
        public:
            Cylinder_Segment(std::vector<std::unique_ptr<Plane_Segment>>& planeGrid, const bool* activated_mask, const int cellActivatedCount);
            ~Cylinder_Segment();
            Cylinder_Segment(Cylinder_Segment& seg, int subRegionId);

        public:
            int get_segment_count() const { return segmentCount; };
            double get_MSE_at(int index) const { return MSE[index]; };
            bool get_inlier_at (int indexA, int indexB) { return inliers[indexA](indexB); };
            int get_local_to_global_mapping(int index) { return local2globalMap[index]; };
            const Eigen::Vector3d& get_axis1_point(int index) const { return pointsAxis1[index];};
            const Eigen::Vector3d& get_axis2_point(int index) const { return pointsAxis2[index];};
            double get_axis_normal(int index) const { return normalsAxis1Axis2[index]; };
            double get_radius(int index) const { return radius[index]; };

            void set_cylindrical_mask(int index, bool val) { cylindricalMask[index] = val; };
            double* get_axis() {return axis;};


        protected:
            double distance(const Eigen::Vector3d& point, int segmentId);

        private:
            double axis[3];

            std::vector<Eigen::MatrixXd> centers;
            std::vector<Eigen::Vector3d> pointsAxis1;
            std::vector<Eigen::Vector3d> pointsAxis2;
            std::vector<double> normalsAxis1Axis2;
            std::vector<MatrixXb> inliers;

            std::vector<double> MSE;
            std::vector<double> radius;
            std::vector<bool> cylindricalMask;

            int segmentCount;
            int* local2globalMap;
    };

}


#endif
