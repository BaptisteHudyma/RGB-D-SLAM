#ifndef RGBDSLAM_UTILS_POSE_HPP
#define RGBDSLAM_UTILS_POSE_HPP

#include "tracking/extended_kalman_filter.hpp"
#include "types.hpp"

#include <Eigen/src/Core/Matrix.h>

#include <memory>

namespace rgbd_slam::utils {

/**
 * \brief Store a position
 */
class PoseBase
{
  public:
    PoseBase();
    PoseBase(const vector3& position, const quaternion& rotation);
    PoseBase(const vector7& vector);

    virtual ~PoseBase() = default;

    // setters
    void set_parameters(const vector3& position, const quaternion& rotation) noexcept;

    // getters
    [[nodiscard]] vector3 get_position() const noexcept { return _position; }
    [[nodiscard]] matrix33 get_rotation_matrix() const noexcept { return _rotation.toRotationMatrix(); }
    [[nodiscard]] quaternion get_rotation_quaternion() const noexcept { return _rotation; }
    /**
     * \return a 6 element vector of the position followed by the rotation in radians
     */
    [[nodiscard]] vector6 get_vector_euler() const noexcept
    {
        vector6 t;
        t << _position, _rotation.toRotationMatrix().eulerAngles(0, 1, 2);
        return t;
    }

    static constexpr size_t positionIndex = 0;
    static constexpr size_t positionSize = 3;

    static constexpr size_t rotationIndex = positionIndex + positionSize;
    static constexpr size_t rotationSize = 4;

    /**
     * \brief get a vector representation of this state
     */
    [[nodiscard]] vector7 get_vector() const noexcept;

    /**
     * \brief Set the state from a vector representation
     */
    void set_from_vector(const vector7& vec);

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
    quaternion _rotation;
    vector3 _position;
};

/**
 * \brief A pose space with associated speed
 */
class PoseBaseWithSpeed : public PoseBase
{
  public:
    PoseBaseWithSpeed();
    PoseBaseWithSpeed(const vector3& position, const quaternion& rotation);
    PoseBaseWithSpeed(const Eigen::Vector<double, 13>& vector);

    void update_position_speed(vector3 positionSpeed) { _positionSpeed = positionSpeed; }
    void update_rotation_speed(vector3 rotationSpeed) { _rotationSpeed = rotationSpeed; }

    vector3 get_position_speed() const { return _positionSpeed; }
    vector3 get_rotation_speed() const { return _rotationSpeed; }

    static constexpr size_t positionSpeedIndex = rotationIndex + rotationSize;
    static constexpr size_t positionSpeedSize = 3;

    static constexpr size_t rotationSpeedIndex = positionSpeedIndex + positionSpeedSize;
    static constexpr size_t rotationSpeedSize = 3;

    [[nodiscard]] Eigen::Vector<double, 13> get_vector() const noexcept
    {
        Eigen::Vector<double, 13> t;
        t.segment<rotationSize + positionSize>(positionIndex) = PoseBase::get_vector();
        t.segment<positionSpeedSize>(positionSpeedIndex) = _positionSpeed;
        t.segment<rotationSpeedSize>(rotationSpeedIndex) = _rotationSpeed;
        return t;
    }
    void set_from_vector(const Eigen::Vector<double, 13>& vec)
    {
        PoseBase::set_from_vector(vec.segment<rotationSize + positionSize>(positionIndex));
        _positionSpeed = vec.segment<positionSpeedSize>(positionSpeedIndex);
        _rotationSpeed = vec.segment<rotationSpeedSize>(rotationSpeedIndex);
    }

    /**
     * \brief Predict the next state using the given delta time since last observation
     */
    PoseBaseWithSpeed predict(const double deltaTime) const;
    PoseBaseWithSpeed predict(const double deltaTime, Eigen::Matrix<double, 13, 13>& jacobian) const;

  private:
    // track speed
    vector3 _rotationSpeed;
    vector3 _positionSpeed;
};

/**
 * \brief Store a position with variance estimations
 */
class Pose : public PoseBaseWithSpeed
{
  public:
    using PoseVarianceT = Eigen::Matrix<double, 13, 13>;

    Pose();
    Pose(const vector3& position, const quaternion& rotation);
    Pose(const vector3& position, const quaternion& rotation, const PoseVarianceT& poseVariance);
    Pose(const Eigen::Vector<double, 13>& vector);

    /**
     * \reset this pose, as if starting from a new world
     */
    void reset_new_world(const double updateTime_s = 0.0);

    /**
     * \brief Return the pose variance for position and rotation
     */
    [[nodiscard]] matrix66 get_pose_variance() const noexcept;
    [[nodiscard]] matrix77 get_pose_variance_quaternion() const noexcept;

    PoseVarianceT get_covar() const { return _poseVariance; }
    void set_position_variance(const matrix77& poseVar) { _poseVariance.block<7, 7>(0, 0) = poseVar; }

    Pose predict(const double measurmentTime_s) const noexcept;

    /**
     * \brief Update this pose using a new pose measurment
     * \param[in] measurment A new pose measurment
     * \param[in] measurmentCovariance The covariance of the new pose
     * \param[in] measurmentTime_s The time of the measurment
     *
     * \return True if the update process succeeded. False if nothing changed
     */
    bool update_with_new_pose(const PoseBase& measurment,
                              const matrix77& measurmentCovariance,
                              const double measurmentTime_s);

    /**
     * \brief A display function, to avoid a friend operator function
     */
    void display(std::ostream& os) const noexcept override;

    double get_update_time() const { return _latestUpdateTime_s; }

    // error size
    static constexpr size_t linearErrorIndex = 0;
    static constexpr size_t linearErrorSize = 3;

    static constexpr size_t rotationErrorIndex = linearErrorIndex + linearErrorSize;
    static constexpr size_t rotationErrorSize = 4;

    static constexpr size_t linearSpeedErrorIndex = rotationErrorIndex + rotationErrorSize;
    static constexpr size_t linearSpeedErrorSize = 3;

    static constexpr size_t angularSpeedErrorIndex = linearSpeedErrorIndex + linearSpeedErrorSize;
    static constexpr size_t angularSpeedErrorSize = 3;

  protected:
    void build_pose_tracker() const;
    PoseVarianceT get_Q(double deltaTime) const;

  private:
    /// time of the latest update operation
    double _latestUpdateTime_s = 0.0;

    // pose variance is represented as :
    // - 3 parameters for position (xyz)
    // - 4 parameters for rotation (quaternion wxyz)
    // - 3 parameters for position speed
    // - 3 parameters for rotation speed
    PoseVarianceT _poseVariance = PoseVarianceT::Zero();

    inline static std::unique_ptr<tracking::ExtendedKalmanFilter<13, 7>> _poseKalman = nullptr;
};

std::ostream& operator<<(std::ostream& os, const PoseBase& pose);
std::ostream& operator<<(std::ostream& os, const Pose& pose);

// array of poses
using pose_array = std::vector<Pose, Eigen::aligned_allocator<Pose>>;

} // namespace rgbd_slam::utils

#endif
