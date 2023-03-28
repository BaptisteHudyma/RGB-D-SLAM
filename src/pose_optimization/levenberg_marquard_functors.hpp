#ifndef RGBDSLAM_POSEOPTIMIZATION_LMFUNCTORS_HPP
#define RGBDSLAM_POSEOPTIMIZATION_LMFUNCTORS_HPP

#include "../types.hpp"
#include "../matches_containers.hpp"
#include "../utils/pose.hpp"

// types
#include <unsupported/Eigen/NonLinearOptimization>

namespace rgbd_slam::pose_optimization {

/**
 * \brief Structure given to the Levenberg-Marquardt algorithm. It optimizes a rotation (quaternion) and a translation
 * (vector3) using the matched features from a frame to the local map, using their distances to one another as the main
 * metric.
 */
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic> struct Levenberg_Marquardt_Functor
{
    // tell the called the numerical type and input/ouput size
    using Scalar = _Scalar;
    enum
    {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };

    // typedefs for the original functor
    using InputType = Eigen::Matrix<Scalar, InputsAtCompileTime, 1>;
    using ValueType = Eigen::Matrix<Scalar, ValuesAtCompileTime, 1>;
    using JacobianType = Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime>;

    Levenberg_Marquardt_Functor(const uint inputCount, const uint outputCount) :
        _inputCount(inputCount),
        _outputCount(outputCount)
    {
    }

    uint values() const { return _outputCount; }
    uint inputs() const { return _inputCount; }

    uint _inputCount;
    uint _outputCount;
};

/**
 * \brief Compute a Lie projection of this quaternion for optimization purposes (Scaled Axis representation)
 * \param[in] quat The quaternion to transform
 * \return The coefficients corresponding to this quaternion
 */
vector3 get_scaled_axis_coefficients_from_quaternion(const quaternion& quat);

/**
 * \brief Compute a quaternion from the Lie projection (Scaled Axis representation)
 * \param[in] optimizationCoefficients The coefficients to transform back to quaternion
 * \return The quaternion obtained from the coefficients
 */
quaternion get_quaternion_from_scale_axis_coefficients(const vector3& optimizationCoefficients);

/**
 * \brief Implementation of the main pose and orientation optimisation, to be used by the Levenberg Marquard
 * optimisator.
 */
struct Global_Pose_Estimator : Levenberg_Marquardt_Functor<double>
{
    // Simple constructor
    /**
     * \param[in] points Matched 2D (screen) to 3D (world) points
     * \param[in] planes Matched camera to world planes
     */
    Global_Pose_Estimator(const matches_containers::match_point_container& points,
                          const matches_containers::match_plane_container& planes);

    /**
     * \brief Implementation of the objective function
     *
     * \param[in] optimizedParameters The vector of parameters to optimize (Size M)
     * \param[out] outputScores The vector of errors, of size N (N the number of points)
     */
    int operator()(const vectorxd& optimizedParameters, vectorxd& outputScores) const;

  private:
    const matches_containers::match_point_container& _points;
    const matches_containers::match_plane_container& _planes;
    std::vector<double> _dividers; // contains the divider of every output coefficients
};

struct Global_Pose_Functor : Eigen::NumericalDiff<Global_Pose_Estimator>
{
};

/**
 * \brief Use for debug.
 * \param[in] status The status to converto string
 * \return Returns a string with the human readable version of Eigen LevenbergMarquardt output status
 */
std::string get_human_readable_end_message(Eigen::LevenbergMarquardtSpace::Status status);

} // namespace rgbd_slam::pose_optimization

#endif
