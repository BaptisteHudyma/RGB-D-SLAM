#ifndef RGBDSLAM_FEATURES_PRIMITIVES_PLANESEGMENT_HPP
#define RGBDSLAM_FEATURES_PRIMITIVES_PLANESEGMENT_HPP

#include <Eigen/Dense>
#include <memory>
#include "PlaneSegment.hpp"
#include "types.hpp"


namespace rgbd_slam {
namespace features {
namespace primitives {

    /**
     * \brief Node class representing a depth graph point. Used to find planes in a depth image. Mainly inspired from CAPE program
     */
    class Plane_Segment {
        public:
            /**
             * \brief Initialize the plane segment with the points from the depth matrix
             *
             * \param[in] cellWidth Width and height of the depth image divisions
             * \param[in] ptsPerCellCount
             */
            Plane_Segment(const uint cellWidth, const uint ptsPerCellCount);
            Plane_Segment(const Plane_Segment& seg);

            void init_plane_segment(const Eigen::MatrixXf& depthCloudArray, const uint cellId);


            /**
             * \return True if this plane segment presents a depth discontinuity with another one.
             */
            bool is_depth_discontinuous(const Plane_Segment& planeSegment);

            /**
             * \brief Merge the PCA saved values in prevision of a plane fitting. This function do not make any new plane calculations
             *
             * \param[in] planeSegment Another plane segment
             */
            void expand_segment(const Plane_Segment& planeSegment);

            /**
             * \brief Merge the PCA saved values in prevision of a plane fitting. This function do not make any new plane calculations
             *
             * \param[in] planeSegment Another plane segment
             */
            void expand_segment(const std::unique_ptr<Plane_Segment>& planeSegment);

            /**
             * \brief Compute the dot product of two plane normals
             *
             * \return A number between -1 and 1
             */
            double get_normal_similarity(const Plane_Segment& p);

            /**
             * \brief Compute the signed distance from a plane to a point
             */
            double get_signed_distance(const double point[3]);

            /**
             * \brief Compute the signed distance from a plane to a point
             */
            double get_signed_distance(const vector3& point);

            /**
             * \brief Fit a plane to the contained points using PCA
             */
            void fit_plane();   //fit a plane to this node points

            /**
              * \brief Clears this segment parameters to restart analysis
              */
            void clear_plane_parameters();    //clear node plane parameters  

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        public: //getters
                double get_MSE() const { return _MSE; };
                const vector3& get_normal() const { return _normal; };
                const vector3& get_mean() const { return _mean; };
                double get_plane_d() const { return _d; };
                bool is_planar() const { return _isPlanar; };
                double get_score() const { return _score; };


        protected:

        private:
                const uint _ptsPerCellCount;  //max nb of points per initial cell
                const uint _minZeroPointCount;  //min acceptable zero points in a node
                const uint _cellWidth;
                const uint _cellHeight;

                uint _pointCount;         //point count
                double _score;    //plane fitting score
                double _MSE;     //plane fitting mean square error
                bool _isPlanar;  //true if node represent a correct node, false: ignore node while mapping

                vector3 _mean;     //mean point of all points in node
                vector3 _normal;   //fitted plane normal
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
}
}

#endif
