#ifndef PLANE_SEGMENT_H
#define PLANE_SEGMENT_H 

#include <Eigen/Dense>
#include <memory>
#include "PlaneSegment.hpp"


namespace primitiveDetection {

    /*
     *  Node class representing a depth graph point
     *  Used to find planes in a depth image
     *
     *  Mainly inspired from CAPE program
     *
     */
    class Plane_Segment {
        public:
            Plane_Segment(unsigned int cellWidth, unsigned int ptsPerCellCount);
            Plane_Segment(const Plane_Segment& seg);

            void init_plane_segment(const Eigen::MatrixXf& depthCloudArray, unsigned int cellId);

            bool is_depth_discontinuous(const Plane_Segment& planeSegment);
            void expand_segment(const Plane_Segment& planeSegment);
            void expand_segment(const std::unique_ptr<Plane_Segment>& planeSegment);

            double get_normal_similarity(const Plane_Segment& p);
            double get_signed_distance(const double point[3]);
            double get_signed_distance(const Eigen::Vector3d& point);

            void fit_plane();   //fit a plane to this node points
            void clear_plane_parameters();    //clear node plane parameters  

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        public: //getters
            double get_MSE() const { return _MSE; };
            const Eigen::Vector3d& get_normal() const { return _normal; };
            const Eigen::Vector3d& get_mean() const { return _mean; };
            double get_plane_d() const { return _d; };
            bool is_planar() const { return _isPlanar; };
            double get_score() const { return _score; };


        protected:

        private:
            const unsigned int _ptsPerCellCount;  //max nb of points per initial cell
            const unsigned int _minZeroPointCount;  //min acceptable zero points in a node
            const unsigned int _cellWidth;
            const unsigned int _cellHeight;

            unsigned int _pointCount;         //point count
            double _score;    //plane fitting score
            double _MSE;     //plane fitting mean square error
            bool _isPlanar;  //true if node represent a correct node, false: ignore node while mapping

            Eigen::Vector3d _mean;     //mean point of all points in node
            Eigen::Vector3d _normal;   //fitted plane normal
            double _d;           //fitted plane d param (ax + by + xz + d)

        private:

            //PCA stored coeffs: efficient calculations of point cloud characteristics
            double _Sx;      //sum of x
            double _Sy;      //sum of y
            double _Sz;      //sum of z
            double _Sxs;     //sum of x squared
            double _Sys;     //sum of y squared
            double _Szs;     //sum of z squared
            double _Sxy;     //sum of x*y
            double _Syz;     //sum of y*z
            double _Szx;     //sum of z*x

        private:
            //prevent backend copy
            Plane_Segment& operator=(const Plane_Segment& seg);

    };

}

#endif
