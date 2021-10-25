#ifndef POSE_CONTAINER_HPP
#define POSE_CONTAINER_HPP

#include "types.hpp"

#include <Eigen/Dense>
#include <Eigen/StdVector>

namespace rgbd_slam { 
namespace utils {

    /**
     * \brief Store a position
     */
    class Pose {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        public:
            Pose(); 
            Pose(const vector3 &position, const quaternion &orientation);

            //setters
            void set_parameters(const vector3 &position, const quaternion &orientation);

            void update(const vector3& position, const quaternion& orientation);

            //getters
            const vector3 get_position() const { return _position; }
            const matrix33 get_orientation_matrix() const { return _orientation.toRotationMatrix(); }
            const quaternion get_orientation_quaternion() const { return _orientation; }

            /**
             * \brief A display function, to avoid a friend operator function
             */
            void display(std::ostream& os) const;

        private:
            quaternion _orientation;
            vector3 _position;
    };

    std::ostream& operator<<(std::ostream& os, const Pose& pose);

    //array of poses
    typedef std::vector<Pose, Eigen::aligned_allocator<Pose>> pose_array;

}
}


#endif
