#ifndef RGBDSLAM_UTILS_POSE_HPP
#define RGBDSLAM_UTILS_POSE_HPP

#include "../types.hpp"

namespace rgbd_slam { 
    namespace utils {

        /**
         * \brief Store a position
         */
        class PoseBase {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            public:
                PoseBase(); 
                PoseBase(const vector3 &position, const quaternion &orientation);

                //setters
                void set_parameters(const vector3 &position, const quaternion &orientation);

                void update(const vector3& position, const quaternion& orientation);

                //getters
                const vector3 get_position() const { return _position; }
                const matrix33 get_orientation_matrix() const { return _orientation.toRotationMatrix(); }
                const quaternion get_orientation_quaternion() const { return _orientation; }

                /**
                 * \brief compute a position error (Units are the same as the position units)
                 */
                double get_position_error(const PoseBase& pose) const;
                /**
                 * \brief compute a rotation error (degrees)
                 */
                double get_rotation_error(const PoseBase& pose) const;

                /**
                 * \brief A display function, to avoid a friend operator function
                 */
                void display(std::ostream& os) const;

            private:
                quaternion _orientation;
                vector3 _position;
        };

        /**
         * \brief Store a position with variance estimations
         */
        class Pose :
            public PoseBase
        {
            public:
                Pose(); 
                Pose(const vector3& position, const quaternion& orientation);
                Pose(const vector3& position, const quaternion& orientation, const vector3& poseVariance);

                void set_position_variance(const vector3& variance) { _positionVariance = variance; };
                const vector3 get_position_variance() const { return _positionVariance; };

                /**
                 * \brief A display function, to avoid a friend operator function
                 */
                void display(std::ostream& os) const;

            private:
                vector3 _positionVariance;
        };

        std::ostream& operator<<(std::ostream& os, const PoseBase& pose);
        std::ostream& operator<<(std::ostream& os, const Pose& pose);

        //array of poses
        typedef std::vector<Pose, Eigen::aligned_allocator<Pose>> pose_array;

    }
}


#endif
