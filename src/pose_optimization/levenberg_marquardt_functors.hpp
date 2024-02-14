#ifndef RGBDSLAM_POSEOPTIMIZATION_LMFUNCTORS_HPP
#define RGBDSLAM_POSEOPTIMIZATION_LMFUNCTORS_HPP

#include "pose.hpp"
#include "types.hpp"
#include "matches_containers.hpp"

// types
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/src/NumericalDiff/NumericalDiff.h>

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
 * \param[in] quat The pose to transform
 * \return The coefficients corresponding to this pose
 */
vector6 get_optimization_coefficient_from_pose(const utils::PoseBase& pose);

/**
 * \brief Compute a quaternion from the Lie projection (Scaled Axis representation)
 * \param[in] optimizationCoefficients The coefficients to transform back to pose
 * \return The pose obtained from the coefficients
 */
utils::PoseBase get_pose_from_optimization_coefficients(const vector6& optimizationCoefficients);
utils::PoseBase get_pose_from_optimization_coefficients(const vector6& optimizationCoefficients,
                                                        Eigen::Matrix<double, 7, 6>& jacobian);

/**
 * \brief Implementation of the main pose and orientation optimisation, to be used by the Levenberg Marquard
 * optimisator.
 */
struct Global_Pose_Estimator : Levenberg_Marquardt_Functor<double>
{
    // Simple constructor
    /**
     * \param[in] optimizationParts The sum of all feature parts
     * \param[in] features The container for the matched features
     */
    Global_Pose_Estimator(const size_t optimizationParts, const matches_containers::match_container* const features);

    /**
     * \brief Implementation of the objective function
     *
     * \param[in] optimizedParameters The vector of parameters to optimize (Size M)
     * \param[out] outputScores The vector of errors, of size N (N is optimizationParts)
     */
    int operator()(const vector6& optimizedParameters, vectorxd& outputScores) const;

    /**
     * \brief Compute the covariance of the optimizedParameters, depending on the given features and jacobian of the
     * transformation
     * \param[in] features The set of features that the optimization was made on
     * \param[in] optimizedPose The optimized pose obtained by the parameters that best fit the model
     * \param[in] jacobian The jacobian of the optimization process
     * \param[out] inputCovariance The covariance of the input parameters
     *
     * \return True if a correct covariance could be found
     */
    [[nodiscard]] static bool get_input_covariance(const matches_containers::match_container& features,
                                                   const utils::PoseBase& optimizedPose,
                                                   const matrixd& jacobian,
                                                   matrix66& inputCovariance) noexcept;

  private:
    // use pointers to prevent useless copy
    const size_t _optimizationParts;
    const matches_containers::match_container* const _features;
};

struct Global_Pose_Functor : Eigen::NumericalDiff<Global_Pose_Estimator, Eigen::Central>
{
};

/**
 * \brief Use for debug.
 * \param[in] status The status to converto string
 * \return Returns a string with the human readable version of Eigen LevenbergMarquardt output status
 */
[[nodiscard]] std::string get_human_readable_end_message(Eigen::LevenbergMarquardtSpace::Status status) noexcept;

} // namespace rgbd_slam::pose_optimization

#endif
