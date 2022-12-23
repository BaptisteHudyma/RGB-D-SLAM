#ifndef RGBDSLAM_FEATURES_PRIMITIVES_PLANESEGMENT_HPP
#define RGBDSLAM_FEATURES_PRIMITIVES_PLANESEGMENT_HPP

#include "../../types.hpp"


namespace rgbd_slam {
    namespace features {
        namespace primitives {

            /**
             * \brief Node class representing a depth graph point.
             * Used to find planes in a depth image.
             * Mainly inspired from CAPE program
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

                    void init_plane_segment(const matrixf& depthCloudArray, const uint cellId);

                    /**
                     * \brief Merge the PCA saved values in prevision of a plane fitting. This function do not make any new plane calculations
                     *
                     * \param[in] planeSegment Another plane segment
                     */
                    void expand_segment(const Plane_Segment& planeSegment);

                    /**
                     * \brief Compute the dot product of two plane normals, It is thecos of the angle between those normals
                     * \param[in] p The other plane segment to check the angle to
                     * \return A number between -1 and 1
                     */
                    double get_cos_angle(const Plane_Segment& p) const;

                    /**
                     * \brief Return the distance from a plane to the given point
                     */
                    double get_point_distance(const vector3& point) const;

                    /**
                     * \param[in] maxMatchDistance Maximum distance after which two planes wont be merged
                     * \return True if the planes could be merged. It is based on the normal angles and center distances
                     */
                    bool can_be_merged(const Plane_Segment& p, const double maxMatchDistance) const;

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
                        vector3 get_normal() const { return _normal; };
                        vector3 get_mean() const { return _mean; };
                        double get_plane_d() const { return _d; };
                        bool is_planar() const { return _isPlanar; };
                        double get_score() const { return _score; };


                protected:
                        bool is_cell_vertical_continuous(const matrixf& depthMatrix) const;
                        bool is_cell_horizontal_continuous(const matrixf& depthMatrix) const;

                private:
                        // TODO: those const members could be static  
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
