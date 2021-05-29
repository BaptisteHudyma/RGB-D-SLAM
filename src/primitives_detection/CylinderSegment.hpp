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
            /**
             * \def Main constructor: fits a cylinder using the plane segments in planeGrid, using RANSAC
             *
             * \param[in] planeGrid The plane segment container
             * \param[in] planeCount
             * \param[in] activated_mask An array of size planeCount, referencing activated plane segments 
             * \param[in] cellActivatedCount
             */
            Cylinder_Segment(const std::unique_ptr<Plane_Segment>* planeGrid, unsigned int planeCount, const bool* activated_mask, unsigned int cellActivatedCount);

            /**
             * \brief Copy constructor
             */
            Cylinder_Segment(const Cylinder_Segment& seg, unsigned int subRegionId);

            /**
             * \brief Copy constructor
             */
            Cylinder_Segment(const Cylinder_Segment& seg);

            /**
             * \brief Compute the point to cylinder surface distance. This distance is an approximation, our cylinder being defined as a sum of plane segments and points on it's main axis
             *
             * \param[in] point The point to compute distance to
             *
             * \return The signed distance between the point and cylinder surface
             */
            double distance(const Eigen::Vector3d& point);

            ~Cylinder_Segment();

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        public:
                /**
                 * \brief 
                 *
                 * \return The number of plane segments fitted in this cylinder surface
                 */
                unsigned int get_segment_count() const;

                /**
                 * \brief 
                 *
                 * \param[in] index The index of the cylinder part to search
                 *
                 * \return the Mean Sqared Error of the fitting process
                 */
                double get_MSE_at(unsigned int index) const;

                /**
                 * \brief
                 *
                 *
                 */
                bool is_inlier_at (unsigned int indexA, unsigned int indexB) const;

                /**
                 * \brief 
                 *
                 * \param[in] index The index of the cylinder part to search
                 *
                 */
                unsigned int get_local_to_global_mapping(unsigned int index) const;

                /**
                 * \brief 
                 *
                 * \param[in] index The index of the cylinder part to search
                 *
                 */
                const Eigen::Vector3d& get_axis1_point(unsigned int index) const;

                /**
                 * \brief 
                 *
                 * \param[in] index The index of the cylinder part to search
                 *
                 */
                const Eigen::Vector3d& get_axis2_point(unsigned int index) const;

                /**
                 * \def Return the normal of a portion of this cylinder segement
                 *
                 * \param[in] index The index of the portion to return, between 0 and _normalsAxis1Axis2.size()
                 */
                double get_axis_normal(unsigned int index) const;


                /**
                 * \def Return the radius of this cylinder segment
                 *
                 * \return The radius of the cylinder segment, in frame units
                 */
                double get_radius(unsigned int index) const;

                /**
                 * \def Return the absolute result of the dot product of the two normals
                 *
                 * \param[in] other The cyclinder segment to compare normal with
                 *
                 * \return A double between 0 and 1, 0 when the normals are orthogonal, 1 il they are parallels.
                 */
                double get_normal_similarity(const Cylinder_Segment& other);



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
                //prevent dangerous and inefficient back end copy
                Cylinder_Segment& operator=(const Cylinder_Segment& seg);   //copy operator
    };


}


#endif
