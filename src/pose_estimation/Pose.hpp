#ifndef POSE_CONTAINER_HPP
#define POSE_CONTAINER_HPP

#include "types.hpp"

#include <Eigen/Dense>
#include <Eigen/StdVector>

namespace poseEstimation {

    /**
     * \brief Store a position
     */
    class Pose {
        public:
            Pose(); 
            Pose(const vector3 &position, const quaternion &orientation);

            //setters
            void set_parameters(const vector3 &position, const quaternion &orientation);

            void update(const vector3& position, const quaternion& orientation);
    
            //getters
            vector3 get_position() const { return _position; }
            matrix33 get_orientation_matrix() const { return _orientation.toRotationMatrix(); }
            quaternion get_orientation_quaternion() const { return _orientation; }

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        private:
                quaternion _orientation;
                vector3 _position;
    };

    //array of poses
    typedef std::vector<Pose, Eigen::aligned_allocator<Pose>> pose_array;
}


#endif
