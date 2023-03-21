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
                 * \return a 6 element vector of the position followed by the rotation in radians
                 */
                const vector6 get_vector() const
                {
                    vector6 t;
                    t << _position, _orientation.toRotationMatrix().eulerAngles(0, 1, 2);
                    return t;
                }

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
                Pose(const vector3& position, const quaternion& orientation, const matrix66& poseVariance);

                void set_position_variance(const matrix66& variance) { _positionVariance = variance; };
                const matrix66 get_pose_variance() const { return _positionVariance; };
                const matrix33 get_position_variance() const { return _positionVariance.block(0, 0, 3, 3); };

                /**
                 * \brief A display function, to avoid a friend operator function
                 */
                void display(std::ostream& os) const;

            private:
                matrix66 _positionVariance;
        };

        std::ostream& operator<<(std::ostream& os, const PoseBase& pose);
        std::ostream& operator<<(std::ostream& os, const Pose& pose);

        //array of poses
        typedef std::vector<Pose, Eigen::aligned_allocator<Pose>> pose_array;

    }
}


#endif
