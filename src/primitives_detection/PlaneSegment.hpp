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
            Plane_Segment(const int cellWidth, const int ptsPerCellCount);
            Plane_Segment(const Plane_Segment& seg);

            void init_plane_segment(const Eigen::MatrixXf& depthCloudArray, const int cellId);

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
            const double get_MSE() const { return MSE; };
            const Eigen::Vector3d& get_normal() const { return normal; };
            const Eigen::Vector3d& get_mean() const { return mean; };
            const double get_plane_d() const { return d; };
            const bool is_planar() const { return isPlanar; };
            const double get_score() const { return score; };


        protected:

        private:
            const int ptsPerCellCount;  //max nb of points per initial cell
            const int minZeroPointCount;  //min acceptable zero points in a node
            const int cellWidth;
            const int cellHeight;

            int pointCount;         //point count
            double score;    //plane fitting score
            double MSE;     //plane fitting mean square error
            bool isPlanar;  //true if node represent a correct node, false: ignore node while mapping

            Eigen::Vector3d mean;     //mean point of all points in node
            Eigen::Vector3d normal;   //fitted plane normal
            double d;           //fitted plane d param (ax + by + xz + d)

        private:

            //PCA stored coeffs: efficient calculations of point cloud characteristics
            double Sx;      //sum of x
            double Sy;      //sum of y
            double Sz;      //sum of z
            double Sxs;     //sum of x squared
            double Sys;     //sum of y squared
            double Szs;     //sum of z squared
            double Sxy;     //sum of x*y
            double Syz;     //sum of y*z
            double Szx;     //sum of z*x

        private:
            //prevent backend copy
            Plane_Segment& operator=(const Plane_Segment& seg);

    };

}

#endif
