#ifndef CYLINDER_SEGMENT_H
#define CYLINDER_SEGMENT_H

#include "PlaneSegment.hpp"
#include "Parameters.hpp"
#include <vector>
#include <memory>
#include <Eigen/Dense>


namespace planeDetection {
    typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>  MatrixXb;


    class Cylinder_Segment {
        public:
            Cylinder_Segment(std::unique_ptr<Plane_Segment>* planeGrid, const unsigned int planeCount, const bool* activated_mask, const unsigned int cellActivatedCount);
            Cylinder_Segment(const Cylinder_Segment& seg, int subRegionId);
            Cylinder_Segment(const Cylinder_Segment& seg);              //copy constructor

            ~Cylinder_Segment();

        public:
            int get_segment_count() const;
            double get_MSE_at(unsigned int index) const;
            bool get_inlier_at (unsigned int indexA, unsigned int indexB) const;
            const unsigned int get_local_to_global_mapping(unsigned int index) const;
            const Eigen::Vector3d& get_axis1_point(unsigned int index) const;
            const Eigen::Vector3d& get_axis2_point(unsigned int index) const;
            double get_axis_normal(unsigned int index) const;
            double get_radius(unsigned int index) const;



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

            unsigned int cellActivatedCount;
            unsigned int segmentCount;
            unsigned int* local2globalMap;

        private:
            //prevent dangerous backend copy
            Cylinder_Segment& operator=(const Cylinder_Segment& seg);   //copy operator
    };


}


#endif
