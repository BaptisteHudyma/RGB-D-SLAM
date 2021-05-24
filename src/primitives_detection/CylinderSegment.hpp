#ifndef CYLINDER_SEGMENT_H
#define CYLINDER_SEGMENT_H

#include "PlaneSegment.hpp"
#include "Parameters.hpp"
#include <vector>
#include <memory>
#include <Eigen/Dense>


namespace primitiveDetection {

    class Cylinder_Segment {
        public:
            Cylinder_Segment(const std::unique_ptr<Plane_Segment>* planeGrid, unsigned int planeCount, const bool* activated_mask, unsigned int cellActivatedCount);
            Cylinder_Segment(const Cylinder_Segment& seg, unsigned int subRegionId);
            Cylinder_Segment(const Cylinder_Segment& seg);              //copy constructor
            
            double distance(const Eigen::Vector3d& point);

            ~Cylinder_Segment();

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        public:
            unsigned int get_segment_count() const;
            double get_MSE_at(unsigned int index) const;
            bool get_inlier_at (unsigned int indexA, unsigned int indexB) const;
            unsigned int get_local_to_global_mapping(unsigned int index) const;
            const Eigen::Vector3d& get_axis1_point(unsigned int index) const;
            const Eigen::Vector3d& get_axis2_point(unsigned int index) const;
            double get_axis_normal(unsigned int index) const;
            double get_radius(unsigned int index) const;



        protected:
            double distance(const Eigen::Vector3d& point, unsigned int segmentId);

        private:
            typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>  MatrixXb;
            typedef std::vector<Eigen::Vector3d> vec3d_vector;

            double _axis[3];

            std::vector<Eigen::MatrixXd> _centers;
            vec3d_vector _pointsAxis1;
            vec3d_vector _pointsAxis2;
            std::vector<double> _normalsAxis1Axis2;
            std::vector<MatrixXb> _inliers;

            std::vector<double> _MSE;
            std::vector<double> _radius;

            unsigned int _cellActivatedCount;
            unsigned int _segmentCount;
            unsigned int* _local2globalMap;

        private:
            //prevent dangerous backend copy
            Cylinder_Segment& operator=(const Cylinder_Segment& seg);   //copy operator
    };


}


#endif
