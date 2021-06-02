#ifndef POSE_HPP
#define POSE_HPP

#include <Eigen/Dense>
#include <Eigen/StdVector>

namespace poseEstimation {

    typedef Eigen::Matrix<double, 2, 1> vector2;
    typedef Eigen::Matrix<double, 3, 1> vector3;
    typedef Eigen::Matrix<double, 4, 1> vector4;
    typedef Eigen::Matrix<double, 3, 3> matrix33;
    typedef Eigen::Matrix<double, 3, 4> matrix34;
    typedef Eigen::Matrix<double, 4, 4> matrix44;
    typedef Eigen::Quaternion<double> quaternion;

    typedef std::vector<vector2, Eigen::aligned_allocator<vector2>> vector2_array;
    typedef std::vector<vector3, Eigen::aligned_allocator<vector3>> vector3_array;
    typedef std::vector<vector4, Eigen::aligned_allocator<vector4>> vector4_array;
    typedef std::vector<matrix33, Eigen::aligned_allocator<matrix33>> matrix33_array;
    typedef std::vector<matrix34, Eigen::aligned_allocator<matrix34>> matrix34_array;
    typedef std::vector<matrix44, Eigen::aligned_allocator<matrix44>> matrix44_array;
    typedef std::vector<quaternion, Eigen::aligned_allocator<quaternion>> quaternion_array;

    /**
     * \brief Store a position
     */
    class Pose {
        public:
            Pose() {
                this->position.setZero();
                this->orientation.setIdentity();
            }
            Pose(const vector3 &position, const quaternion &orientation) {
                set_parameters(position, orientation);
            }

            //setters
            void set_parameters(const vector3 &position, const quaternion &orientation) {
                this->orientation = orientation;
                this->position = position;
            }

            //getters
            vector3 get_position() const { return this->position; }
            matrix33 get_orientation_matrix() const { return this->orientation.toRotationMatrix(); }
            quaternion get_orientation_quaternion() const { return this->orientation; }

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        private:
                quaternion orientation;
                vector3 position;
    };

    //array of poses
    typedef std::vector<Pose, Eigen::aligned_allocator<Pose>> pose_array;
}


#endif
