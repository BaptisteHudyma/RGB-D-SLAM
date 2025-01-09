/**
 * TESTS FOR THE DIFFERENT FEATURE FUSION ALGORITHMS
 *
 * - Point 3D merge with point 3D
 * - Point 3D merge with point 2D
 * - Point inverse depth merge with point 2D
 * - Point inverse depth merge with point 2D
 * - Plane 3D merged with plane 3D
 */

#include "angle_utils.hpp"
#include "camera_transformation.hpp"
#include "coordinates/point_coordinates.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <cstdint>
#include <gtest/gtest.h>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <random>
#include "covariances.hpp"
#include "inverse_depth_with_tracking.hpp"
#include "parameters.hpp"
#include "point_with_tracking.hpp"
#include "types.hpp"

namespace rgbd_slam::tracking {

/**
 * 3D point fusion with 3D observation
 */

/**
 * 3D point fusion with 2D observation
 */

TEST(PointFusion3d, centerPointFusion)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const WorldToCameraMatrix& w2c = utils::compute_world_to_camera_transform(
            utils::compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3::Zero()));

    cv::Mat desc = cv::Mat_<int>::ones(1, 1);
    Point trackPoint(
            WorldCoordinate(0.0, 0.0, 1000.0), WorldCoordinateCovariance {matrix33::Identity() * SQR(0.1)}, desc);

    ScreenCoordinate sc;
    EXPECT_TRUE(trackPoint._coordinates.to_screen_coordinates(w2c, sc));

    // observe point head on, so center of screen
    EXPECT_NEAR(sc.x(), observation.x(), 0.01);
    EXPECT_NEAR(sc.y(), observation.y(), 0.01);
    EXPECT_NEAR(sc.z(), 1000.0, 0.01);

    // track the same observed point, result is still close
    EXPECT_TRUE(trackPoint.track_2d(observation, w2c));

    EXPECT_TRUE(trackPoint._coordinates.to_screen_coordinates(w2c, sc));
    EXPECT_NEAR(sc.x(), observation.x(), 0.01);
    EXPECT_NEAR(sc.y(), observation.y(), 0.01);
    EXPECT_NEAR(sc.z(), 1000.0, 0.01);

    auto pointScreenCovariance3d = utils::propagate_covariance(
            trackPoint._covariance, trackPoint._coordinates.to_screen_coordinates_jacobian(w2c));
    std::cout << pointScreenCovariance3d << std::endl;

    const WorldToCameraMatrix& w2cSideA =
            utils::compute_world_to_camera_transform(utils::compute_camera_to_world_transform_no_correction(
                    utils::get_quaternion_from_euler_angles(EulerAngles(0.0, 90 * EulerToRadian, 0.0)),
                    vector3(-1000.0, 0.0, 1000.0)));

    EXPECT_TRUE(trackPoint.track_2d(observation, w2cSideA));
    // values did not move
    EXPECT_TRUE(trackPoint._coordinates.to_screen_coordinates(w2cSideA, sc));
    EXPECT_NEAR(sc.x(), observation.x(), 0.01);
    EXPECT_NEAR(sc.y(), observation.y(), 0.01);
    EXPECT_NEAR(sc.z(), 1000.0, 0.01);

    pointScreenCovariance3d = utils::propagate_covariance(
            trackPoint._covariance, trackPoint._coordinates.to_screen_coordinates_jacobian(w2cSideA));

    std::cout << pointScreenCovariance3d << std::endl;

    const WorldToCameraMatrix& w2cSideB =
            utils::compute_world_to_camera_transform(utils::compute_camera_to_world_transform_no_correction(
                    utils::get_quaternion_from_euler_angles(EulerAngles(0.0, -90 * EulerToRadian, 0.0)),
                    vector3(1000.0, 0.0, 1000.0)));

    EXPECT_TRUE(trackPoint.track_2d(observation, w2cSideB));
    // values did not move
    EXPECT_TRUE(trackPoint._coordinates.to_screen_coordinates(w2cSideB, sc));
    EXPECT_NEAR(sc.x(), observation.x(), 0.01);
    EXPECT_NEAR(sc.y(), observation.y(), 0.01);
    EXPECT_NEAR(sc.z(), 1000.0, 0.01);

    pointScreenCovariance3d = utils::propagate_covariance(
            trackPoint._covariance, trackPoint._coordinates.to_screen_coordinates_jacobian(w2cSideB));

    std::cout << pointScreenCovariance3d << std::endl;
}

/**
 * Inverse depth point fusion with 2D coordinates
 */

static constexpr double linearityThreshold = 0.1;

void assert_inverse_point_back_proj(const CameraToWorldMatrix& c2w, const ScreenCoordinate2D& observation)
{
    const auto w2c = utils::compute_world_to_camera_transform(c2w);

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);

    const auto screenProjNoMovs = inverseDepth.get_projected_screen_estimation(w2c, 0.0);
    // 1 px error
    EXPECT_NEAR(screenProjNoMovs.x(), observation.x(), 0.1);
    EXPECT_NEAR(screenProjNoMovs.y(), observation.y(), 0.1);

    // retroproject to screen
    ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_world_coordinates().to_screen_coordinates(w2c, screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), observation.x(), 0.01);
    EXPECT_NEAR(screenCoordinates.y(), observation.y(), 0.01);
}

TEST(InverseDepthPointFusion, centerPointParallelFusion)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());

    // convert to inverse
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix33::Identity(), cv::Mat());

    // check projection/backprojection
    assert_inverse_point_back_proj(c2w, observation);

    /**
     ** add a new measurment at the same position
     */

    // linearity should be bad
    EXPECT_GT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);

    // fuse the two points
    EXPECT_TRUE(inverseDepth.track(observation, c2w, matrix33::Identity(), cv::Mat()));
}

TEST(InverseDepthPointFusion, centerPointForwardParallelFusion)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());

    // convert to inverse
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix33::Identity(), cv::Mat());

    // check projection/backprojection
    assert_inverse_point_back_proj(c2w, observation);

    /**
     ** add a new measurment just forward or the position
     * Observation point did not move so it must be far away
     */

    const CameraToWorldMatrix& c2wForward = utils::compute_camera_to_world_transform(
            utils::get_quaternion_from_euler_angles(EulerAngles(0.0, 0.0, 0.0)), vector3(1000, 0.0, 0.0));
    assert_inverse_point_back_proj(c2wForward, observation);

    // fuse the two points
    EXPECT_TRUE(inverseDepth.track(observation, c2wForward, matrix33::Identity(), cv::Mat()));

    // linearity should be bad
    EXPECT_GT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);

    // final depth estimation is pushed far away
    EXPECT_LT(inverseDepth._coordinates.get_inverse_depth(), parameters::detection::inverseDepthBaseline);
}

TEST(InverseDepthPointFusion, centerPointBackwardParallelFusion)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());

    // convert to inverse
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix33::Identity(), cv::Mat());

    // check projection/backprojection
    assert_inverse_point_back_proj(c2w, observation);

    /**
     ** add a new measurment just forward or the position
     * Observation point did not move so it must be far away
     */

    const CameraToWorldMatrix& c2wForward = utils::compute_camera_to_world_transform(
            utils::get_quaternion_from_euler_angles(EulerAngles(0.0, 0.0, 0.0)), vector3(-1000, 0.0, 0.0));
    assert_inverse_point_back_proj(c2wForward, observation);

    // fuse the two points
    EXPECT_TRUE(inverseDepth.track(observation, c2wForward, matrix33::Identity(), cv::Mat()));

    // linearity should be bad
    EXPECT_GT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);

    // final depth estimation is pushed far away
    EXPECT_LT(inverseDepth._coordinates.get_inverse_depth(), parameters::detection::inverseDepthBaseline);
}

TEST(InverseDepthPointFusion, topLeftPointForwardParallelFusion)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(0.0, 0.0);
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());

    // convert to inverse
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix33::Identity(), cv::Mat());

    // check projection/backprojection
    assert_inverse_point_back_proj(c2w, observation);

    /**
     ** add a new measurment just forward or the position
     * Observation point did not move so it must be far away
     */

    const CameraToWorldMatrix& c2wForward = utils::compute_camera_to_world_transform(
            utils::get_quaternion_from_euler_angles(EulerAngles(0.0, 0.0, 0.0)), vector3(1000, 0.0, 0.0));
    assert_inverse_point_back_proj(c2wForward, observation);

    // fuse the two points
    EXPECT_TRUE(inverseDepth.track(observation, c2wForward, matrix33::Identity(), cv::Mat()));

    // linearity should be bad
    EXPECT_GT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);

    // final depth estimation is pushed far away
    EXPECT_LT(inverseDepth._coordinates.get_inverse_depth(), parameters::detection::inverseDepthBaseline);
}

TEST(InverseDepthPointFusion, topLeftPointBackwardParallelFusion)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(0.0, 0.0);
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());

    // convert to inverse
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix33::Identity(), cv::Mat());

    // check projection/backprojection
    assert_inverse_point_back_proj(c2w, observation);

    /**
     ** add a new measurment just forward or the position
     * Observation point did not move so it must be far away
     */

    const CameraToWorldMatrix& c2wForward = utils::compute_camera_to_world_transform(
            utils::get_quaternion_from_euler_angles(EulerAngles(0.0, 0.0, 0.0)), vector3(-1000, 0.0, 0.0));
    assert_inverse_point_back_proj(c2wForward, observation);

    // fuse the two points
    EXPECT_TRUE(inverseDepth.track(observation, c2wForward, matrix33::Identity(), cv::Mat()));

    // linearity should be bad
    EXPECT_GT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);

    // final depth estimation is pushed far away
    EXPECT_LT(inverseDepth._coordinates.get_inverse_depth(), parameters::detection::inverseDepthBaseline);
}

TEST(InverseDepthPointFusion, centerPointFusionFromSide)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());

    // convert to inverse
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix33::Identity() * 0.01, cv::Mat());

    // check projection/backprojection
    assert_inverse_point_back_proj(c2w, observation);

    // check that the projected segment is in fact a point
    utils::Segment<2> originalSegment;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), originalSegment));
    EXPECT_NEAR(originalSegment.get_start_point().x(), observation.x(), 1);
    EXPECT_NEAR(originalSegment.get_start_point().y(), observation.y(), 1);
    EXPECT_NEAR(originalSegment.get_end_point().x(), observation.x(), 1);
    EXPECT_NEAR(originalSegment.get_end_point().y(), observation.y(), 1);

    EXPECT_NEAR(originalSegment.get_end_point().x(), originalSegment.get_start_point().x(), 1);
    EXPECT_NEAR(originalSegment.get_end_point().y(), originalSegment.get_start_point().y(), 1);

    /**
     ** add a new measurment at 90° from the position, further on the trajectory
     */

    const CameraToWorldMatrix& c2wSide90 = utils::compute_camera_to_world_transform(
            utils::get_quaternion_from_euler_angles(EulerAngles(0.0, -90 * EulerToRadian, 0.0)),
            vector3(1000.0, 0.0, 1000.0));
    assert_inverse_point_back_proj(c2wSide90, observation);

    // fuse the two points (multiple observations, process is slow)
    for (uint i = 0; i < 10; i++)
    {
        EXPECT_TRUE(inverseDepth.track(observation, c2wSide90, matrix33::Identity(), cv::Mat()));

        utils::Segment<2> screenSegment;
        EXPECT_TRUE(
                inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2wSide90), screenSegment));
        std::cout << screenSegment.get_start_point().transpose() << "  ----   "
                  << screenSegment.get_end_point().transpose() << std::endl;

        // this is a line, should always be a line throught center of the "screen"
        EXPECT_NEAR(screenSegment.get_start_point().y(), observation.y(), 1);
        EXPECT_NEAR(screenSegment.get_end_point().y(), observation.y(), 1);

        // this projection gives back the depth in screen space, check that it is close to expected
        const auto finalPoint = inverseDepth._coordinates.to_world_coordinates();
        ScreenCoordinate sc;
        EXPECT_TRUE(finalPoint.to_screen_coordinates(utils::compute_world_to_camera_transform(c2wSide90), sc));

        // x has too much uncertainty during opti process, do not check it
        EXPECT_NEAR(sc.y(), observation.y(), 1);
        EXPECT_NEAR(sc.z(), 1000, 10);
    }

    // check that the final projection line is close around the target
    utils::Segment<2> screenSegment;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2wSide90), screenSegment));
    EXPECT_NEAR(screenSegment.get_start_point().x(), observation.x(), 5);
    EXPECT_NEAR(screenSegment.get_start_point().y(), observation.y(), 5);
    EXPECT_NEAR(screenSegment.get_end_point().x(), observation.x(), 5);
    EXPECT_NEAR(screenSegment.get_end_point().y(), observation.y(), 5);

    const auto finalPoint = inverseDepth._coordinates.to_world_coordinates();
    const auto finalPointCovariance = tracking::PointInverseDepth::compute_cartesian_covariance(
            inverseDepth._coordinates, inverseDepth._covariance);
    EXPECT_TRUE(utils::is_covariance_valid(finalPointCovariance));

    // final pose is triangulated
    EXPECT_GT(inverseDepth._coordinates.get_inverse_depth(), parameters::detection::inverseDepthBaseline);

    EXPECT_NEAR(finalPoint.x(), 1000, 0.001);
    // 1cm tolerance
    EXPECT_NEAR(finalPoint.y(), 0, 10);
    EXPECT_NEAR(finalPoint.z(), 0, 10);

    // linearity should be pretty good
    EXPECT_LT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);

    std::cout << finalPointCovariance << std::endl;
}

TEST(InverseDepthPointFusion, centerPointFusionFromOtherSide)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());

    // convert to inverse
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix33::Identity() * 0.01, cv::Mat());

    // check projection/backprojection
    assert_inverse_point_back_proj(c2w, observation);

    // check that the projected segment is in fact a point
    utils::Segment<2> originalSegment;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), originalSegment));
    EXPECT_NEAR(originalSegment.get_start_point().x(), observation.x(), 1);
    EXPECT_NEAR(originalSegment.get_start_point().y(), observation.y(), 1);
    EXPECT_NEAR(originalSegment.get_end_point().x(), observation.x(), 1);
    EXPECT_NEAR(originalSegment.get_end_point().y(), observation.y(), 1);

    EXPECT_NEAR(originalSegment.get_end_point().x(), originalSegment.get_start_point().x(), 1);
    EXPECT_NEAR(originalSegment.get_end_point().y(), originalSegment.get_start_point().y(), 1);

    /**
     ** add a new measurment at 90° from the position, further on the trajectory
     */

    const CameraToWorldMatrix& c2wSide90 = utils::compute_camera_to_world_transform(
            utils::get_quaternion_from_euler_angles(EulerAngles(0.0, 90 * EulerToRadian, 0.0)),
            vector3(-1000.0, 0.0, 1000.0));
    assert_inverse_point_back_proj(c2wSide90, observation);

    // fuse the two points (multiple observations, process is slow)
    for (uint i = 0; i < 10; i++)
    {
        EXPECT_TRUE(inverseDepth.track(observation, c2wSide90, matrix33::Identity(), cv::Mat()));

        utils::Segment<2> screenSegment;
        EXPECT_TRUE(
                inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2wSide90), screenSegment));
        std::cout << screenSegment.get_start_point().transpose() << "  ----   "
                  << screenSegment.get_end_point().transpose() << std::endl;

        // this is a line, should always be a line throught center of the "screen"
        EXPECT_NEAR(screenSegment.get_start_point().y(), observation.y(), 1);
        EXPECT_NEAR(screenSegment.get_end_point().y(), observation.y(), 1);

        // this projection gives back the depth in screen space, check that it is close to expected
        const auto finalPoint = inverseDepth._coordinates.to_world_coordinates();
        ScreenCoordinate sc;
        EXPECT_TRUE(finalPoint.to_screen_coordinates(utils::compute_world_to_camera_transform(c2wSide90), sc));

        // x has too much uncertainty during opti process, do not check it
        EXPECT_NEAR(sc.y(), observation.y(), 1);
        EXPECT_NEAR(sc.z(), 1000, 10);
    }

    // check that the final projection line is close around the target
    utils::Segment<2> screenSegment;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2wSide90), screenSegment));
    EXPECT_NEAR(screenSegment.get_start_point().x(), observation.x(), 5);
    EXPECT_NEAR(screenSegment.get_start_point().y(), observation.y(), 5);
    EXPECT_NEAR(screenSegment.get_end_point().x(), observation.x(), 5);
    EXPECT_NEAR(screenSegment.get_end_point().y(), observation.y(), 5);

    const auto finalPoint = inverseDepth._coordinates.to_world_coordinates();
    const auto finalPointCovariance = tracking::PointInverseDepth::compute_cartesian_covariance(
            inverseDepth._coordinates, inverseDepth._covariance);
    EXPECT_TRUE(utils::is_covariance_valid(finalPointCovariance));

    // final pose is triangulated
    EXPECT_GT(inverseDepth._coordinates.get_inverse_depth(), parameters::detection::inverseDepthBaseline);

    EXPECT_NEAR(finalPoint.x(), 1000, 0.001);
    // 1cm tolerance
    EXPECT_NEAR(finalPoint.y(), 0, 10);
    EXPECT_NEAR(finalPoint.z(), 0, 10);

    // linearity should be pretty good
    EXPECT_LT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);

    std::cout << finalPointCovariance << std::endl;
}

/**
 * Inverse depth point fusion with 3D coordinates
 */

} // namespace rgbd_slam::tracking