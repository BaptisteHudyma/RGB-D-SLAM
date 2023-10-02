#ifndef RGBDSLAM_UTILS_POSE_HPP
#define RGBDSLAM_UTILS_POSE_HPP

#include "../types.hpp"

namespace rgbd_slam::utils {

/**
 * \brief Store a position
 */
class PoseBase
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  public:
    PoseBase();
    PoseBase(const vector3& position, const quaternion& orientation);

    virtual ~PoseBase() = default;

    // setters
    void set_parameters(const vector3& position, const quaternion& orientation) noexcept;

    void update(const vector3& position, const quaternion& orientation) noexcept;

    // getters
    [[nodiscard]] vector3 get_position() const noexcept { return _position; }
    [[nodiscard]] matrix33 get_orientation_matrix() const noexcept { return _orientation.toRotationMatrix(); }
    [[nodiscard]] quaternion get_orientation_quaternion() const noexcept { return _orientation; }
    /**
     * \return a 6 element vector of the position followed by the rotation in radians
     */
    [[nodiscard]] vector6 get_vector() const noexcept
    {
        vector6 t;
        t << _position, _orientation.toRotationMatrix().eulerAngles(0, 1, 2);
        return t;
    }

    /**
     * \brief compute a position error (Units are the same as the position units)
     */
    [[nodiscard]] double get_position_error(const PoseBase& pose) const noexcept;
    /**
     * \brief compute a rotation error (degrees)
     */
    [[nodiscard]] double get_rotation_error(const PoseBase& pose) const noexcept;

    /**
     * \brief A display function, to avoid a friend operator function
     */
    virtual void display(std::ostream& os) const noexcept;

  private:
    quaternion _orientation;
    vector3 _position;
};

/**
 * \brief Store a position with variance estimations
 */
class Pose : public PoseBase
{
  public:
    Pose();
    Pose(const vector3& position, const quaternion& orientation);
    Pose(const vector3& position, const quaternion& orientation, const matrix66& poseVariance);

    virtual ~Pose() = default;

    void set_position_variance(const matrix66& variance) noexcept { _poseVariance = variance; };
    [[nodiscard]] matrix66 get_pose_variance() const noexcept { return _poseVariance; };
    [[nodiscard]] matrix33 get_position_variance() const noexcept { return _poseVariance.block<3, 3>(0, 0); };

    /**
     * \brief A display function, to avoid a friend operator function
     */
    void display(std::ostream& os) const noexcept override;

  private:
    matrix66 _poseVariance;
};

std::ostream& operator<<(std::ostream& os, const PoseBase& pose);
std::ostream& operator<<(std::ostream& os, const Pose& pose);

// array of poses
using pose_array = std::vector<Pose, Eigen::aligned_allocator<Pose>>;

} // namespace rgbd_slam::utils

#endif
