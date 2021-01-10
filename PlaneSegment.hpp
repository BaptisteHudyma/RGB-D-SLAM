#ifndef PLANE_SEGMENT_H
#define PLANE_SEGMENT_H 

#include <Eigen/Dense>
#include <memory>

#define DEPTH_SIGMA_COEFF 1.6e-6
#define DEPTH_SIGMA_MARGIN 4        //[3, 8]
#define DEPTH_DISCONTINUITY_LIMIT 2
#define DEPTH_ALPHA 0.02            //[0.01, 0.02]

namespace planeDetection {

    /*
     *  Node class representing a depth graph point
     *  Used to find planes in a depth image
     *
     *  Mainly inspired from CAPE program
     *
     */
    class Plane_Segment {
        public:
            Plane_Segment(Eigen::MatrixXf& depthCloudArray, int cellId, int ptsPerCellCount, int cellWidth);

            bool is_depth_discontinuous(const Plane_Segment& planeSegment);
            void expand_segment(const Plane_Segment& planeSegment);
            void expand_segment(const std::unique_ptr<Plane_Segment>& planeSegment);

            double get_normal_similarity(const Plane_Segment& p);
            double get_signed_distance(const double point[3]);
            double get_signed_distance(const Eigen::Vector3d& point);

            void fit_plane();   //fit a plane to this node points
            ~Plane_Segment();

        public: //getters
            const double get_MSE() const { return MSE; };
            const Eigen::Vector3d& get_normal() const { return normal; };
            const Eigen::Vector3d& get_mean() const { return mean; };
            const double get_plane_d() const { return d; };
            const bool is_planar() const { return isPlanar; };
            const double get_score() const { return score; };


        protected:
            void clear_plane_parameters();    //clear node plane parameters  

        private:
            int pointCount;         //point count
            int minZeroPointCount;  //min acceptable zero points in a node

            double score;    //plane fitting score
            double MSE;     //plane fitting mean square error
            bool isPlanar;  //true if node represent a correct node, false: ignore node while mapping

            bool hasEmpty;  //stored empty point check
            bool hasDepthDiscontinuity;    //stored depth discontinuity check

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

    };

}

#endif
