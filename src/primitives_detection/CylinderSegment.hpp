#ifndef CYLINDER_SEGMENT_H
#define CYLINDER_SEGMENT_H

#include "PlaneSegment.hpp"
#include "Parameters.hpp"
#include <vector>
#include <memory>
#include <Eigen/Dense>


namespace primitiveDetection {

    typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>  MatrixXb;


    class Cylinder_Segment {
        public:
            Cylinder_Segment(const std::unique_ptr<Plane_Segment>* planeGrid, const unsigned int planeCount, const bool* activated_mask, const unsigned int cellActivatedCount);
            Cylinder_Segment(const Cylinder_Segment& seg, int subRegionId);
            Cylinder_Segment(const Cylinder_Segment& seg);              //copy constructor
            
            double distance(const Eigen::Vector3d& point);

            ~Cylinder_Segment();

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        public:
            int get_segment_count() const;
            double get_MSE_at(const unsigned int index) const;
            bool get_inlier_at (const unsigned int indexA, const unsigned int indexB) const;
            const unsigned int get_local_to_global_mapping(const unsigned int index) const;
            const Eigen::Vector3d& get_axis1_point(const unsigned int index) const;
            const Eigen::Vector3d& get_axis2_point(const unsigned int index) const;
            double get_axis_normal(const unsigned int index) const;
            double get_radius(const unsigned int index) const;



        protected:
            double distance(const Eigen::Vector3d& point, const int segmentId);

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
