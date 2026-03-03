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
#include <gtest/gtest.h>
#include <iostream>
#include <opencv2/line_descriptor/descriptor.hpp>
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
            WorldCoordinate(0.0, 0.0, 1.0), WorldCoordinateCovariance {matrix33::Identity() * SQR(0.0001)}, desc);

    ScreenCoordinate sc;
    EXPECT_TRUE(trackPoint._coordinates.to_screen_coordinates(w2c, sc));

    // observe point head on, so center of screen
    EXPECT_NEAR(sc.x(), observation.x(), 0.01);
    EXPECT_NEAR(sc.y(), observation.y(), 0.01);
    EXPECT_NEAR(sc.z(), 1.0, 0.001);

    // track the same observed point, result is still close
    EXPECT_TRUE(trackPoint.track_2d(observation, w2c, matrix66::Zero()));

    EXPECT_TRUE(trackPoint._coordinates.to_screen_coordinates(w2c, sc));
    EXPECT_NEAR(sc.x(), observation.x(), 0.01);
    EXPECT_NEAR(sc.y(), observation.y(), 0.01);
    EXPECT_NEAR(sc.z(), 1.0, 0.001);

    auto pointScreenCovariance3d = utils::propagate_covariance(
            trackPoint._covariance, trackPoint._coordinates.to_screen_coordinates_jacobian(w2c));
    auto pointStandardDev = pointScreenCovariance3d.diagonal().cwiseSqrt();
    EXPECT_LE(pointStandardDev.x(), 0.1);
    EXPECT_LE(pointStandardDev.y(), 0.1);
    EXPECT_LE(pointStandardDev.z(), 0.1);

    // track point, observe by the side
    const WorldToCameraMatrix& w2cSideA =
            utils::compute_world_to_camera_transform(utils::compute_camera_to_world_transform_no_correction(
                    utils::get_quaternion_from_euler_angles(EulerAngles(0.0, 90 * EulerToRadian, 0.0)),
                    vector3(-1.0, 0.0, 1.0)));

    EXPECT_TRUE(trackPoint.track_2d(observation, w2cSideA, matrix66::Zero()));
    // values did not move
    EXPECT_TRUE(trackPoint._coordinates.to_screen_coordinates(w2cSideA, sc));
    EXPECT_NEAR(sc.x(), observation.x(), 0.01);
    EXPECT_NEAR(sc.y(), observation.y(), 0.01);
    EXPECT_NEAR(sc.z(), 1.0, 0.001);

    pointScreenCovariance3d = utils::propagate_covariance(
            trackPoint._covariance, trackPoint._coordinates.to_screen_coordinates_jacobian(w2cSideA));
    EXPECT_LE(pointStandardDev.x(), 0.06);
    EXPECT_LE(pointStandardDev.y(), 0.06);
    // TODO: this covariance should go lower and lower (triangulation)
    EXPECT_LE(pointStandardDev.z(), 0.1);

    // track point, observe by the other side
    const WorldToCameraMatrix& w2cSideB =
            utils::compute_world_to_camera_transform(utils::compute_camera_to_world_transform_no_correction(
                    utils::get_quaternion_from_euler_angles(EulerAngles(0.0, -90 * EulerToRadian, 0.0)),
                    vector3(1.0, 0.0, 1.0)));

    EXPECT_TRUE(trackPoint.track_2d(observation, w2cSideB, matrix66::Zero()));
    // values did not move
    EXPECT_TRUE(trackPoint._coordinates.to_screen_coordinates(w2cSideB, sc));
    EXPECT_NEAR(sc.x(), observation.x(), 0.01);
    EXPECT_NEAR(sc.y(), observation.y(), 0.01);
    EXPECT_NEAR(sc.z(), 1.0, 0.001);

    pointScreenCovariance3d = utils::propagate_covariance(
            trackPoint._covariance, trackPoint._coordinates.to_screen_coordinates_jacobian(w2cSideB));
    EXPECT_LE(pointStandardDev.x(), 0.06);
    EXPECT_LE(pointStandardDev.y(), 0.06);
    // TODO: this covariance should go lower and lower (triangulation)
    EXPECT_LE(pointStandardDev.z(), 0.1);
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

    const auto screenProjNoMovs = inverseDepth.get_projected_screen_estimation(w2c);
    // 1 px error
    EXPECT_NEAR(screenProjNoMovs.x(), observation.x(), 0.1);
    EXPECT_NEAR(screenProjNoMovs.y(), observation.y(), 0.1);

    // retroproject to screen
    ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_world_coordinates().to_screen_coordinates(w2c, screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), observation.x(), 0.01);
    EXPECT_NEAR(screenCoordinates.y(), observation.y(), 0.01);

    // Check 3D

    const auto screenProj3dNoMovs = inverseDepth.get_projected_screen3d_estimation(w2c);
    // 1 px error
    EXPECT_NEAR(screenProj3dNoMovs.x(), observation.x(), 0.1);
    EXPECT_NEAR(screenProj3dNoMovs.y(), observation.y(), 0.1);

    // retroproject to screen
    ScreenCoordinate screenCoordinates3d;
    WorldCoordinate wc = inverseDepth.to_world_coordinates();
    EXPECT_TRUE(wc.to_screen_coordinates(w2c, screenCoordinates3d));
    // should be the same
    EXPECT_NEAR(screenCoordinates3d.x(), observation.x(), 0.01);
    EXPECT_NEAR(screenCoordinates3d.y(), observation.y(), 0.01);

    EXPECT_NEAR(screenProj3dNoMovs.z(), screenCoordinates3d.z(), 0.5);

    // sanity check: baseline is restored
    EXPECT_NEAR((wc.to_camera_coordinates(w2c)).norm(), 1.0 / parameters::detection::inverseDepthBaseline_m, 0.01);
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
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix66::Zero(), cv::Mat());

    // check projection/backprojection
    assert_inverse_point_back_proj(c2w, observation);

    /**
     ** add a new measurment at the same position
     */

    // linearity should be bad
    EXPECT_GT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);

    // fuse the two points
    EXPECT_TRUE(inverseDepth.track_2D(observation, observation.get_covariance(), c2w, matrix66::Zero(), cv::Mat()));
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
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix66::Zero(), cv::Mat());

    // check projection/backprojection
    assert_inverse_point_back_proj(c2w, observation);

    /**
     ** add a new measurment just forward or the position
     * Observation point did not move so it must be far away
     */

    const CameraToWorldMatrix& c2wForward = utils::compute_camera_to_world_transform(
            utils::get_quaternion_from_euler_angles(EulerAngles(0.0, 0.0, 0.0)), vector3(1.0, 0.0, 0.0));
    assert_inverse_point_back_proj(c2wForward, observation);

    // fuse the two points
    EXPECT_TRUE(
            inverseDepth.track_2D(observation, observation.get_covariance(), c2wForward, matrix66::Zero(), cv::Mat()));

    // linearity should be bad
    EXPECT_GT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);

    // final depth estimation is pushed far away
    EXPECT_LT(inverseDepth._coordinates.get_inverse_depth(), parameters::detection::inverseDepthBaseline_m);
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
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix66::Zero(), cv::Mat());

    // check projection/backprojection
    assert_inverse_point_back_proj(c2w, observation);

    /**
     ** add a new measurment just forward or the position
     * Observation point did not move so it must be far away
     */

    const CameraToWorldMatrix& c2wForward = utils::compute_camera_to_world_transform(
            utils::get_quaternion_from_euler_angles(EulerAngles(0.0, 0.0, 0.0)), vector3(-1.0, 0.0, 0.0));
    assert_inverse_point_back_proj(c2wForward, observation);

    // fuse the two points
    EXPECT_TRUE(
            inverseDepth.track_2D(observation, observation.get_covariance(), c2wForward, matrix66::Zero(), cv::Mat()));

    // linearity should be bad
    EXPECT_GT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);

    // final depth estimation is pushed far away
    EXPECT_LT(inverseDepth._coordinates.get_inverse_depth(), parameters::detection::inverseDepthBaseline_m);
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
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix66::Zero(), cv::Mat());

    // check projection/backprojection
    assert_inverse_point_back_proj(c2w, observation);

    /**
     ** add a new measurment just forward or the position
     * Observation point did not move so it must be far away
     */

    const CameraToWorldMatrix& c2wForward = utils::compute_camera_to_world_transform(
            utils::get_quaternion_from_euler_angles(EulerAngles(0.0, 0.0, 0.0)), vector3(1.0, 0.0, 0.0));
    assert_inverse_point_back_proj(c2wForward, observation);

    // fuse the two points
    EXPECT_TRUE(
            inverseDepth.track_2D(observation, observation.get_covariance(), c2wForward, matrix66::Zero(), cv::Mat()));

    // linearity should be bad
    EXPECT_GT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);

    // final depth estimation is pushed far away
    EXPECT_LT(inverseDepth._coordinates.get_inverse_depth(), parameters::detection::inverseDepthBaseline_m);
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
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix66::Zero(), cv::Mat());

    // check projection/backprojection
    assert_inverse_point_back_proj(c2w, observation);

    /**
     ** add a new measurment just forward or the position
     * Observation point did not move so it must be far away
     */

    const CameraToWorldMatrix& c2wForward = utils::compute_camera_to_world_transform(
            utils::get_quaternion_from_euler_angles(EulerAngles(0.0, 0.0, 0.0)), vector3(-1.0, 0.0, 0.0));
    assert_inverse_point_back_proj(c2wForward, observation);

    // fuse the two points
    EXPECT_TRUE(
            inverseDepth.track_2D(observation, observation.get_covariance(), c2wForward, matrix66::Zero(), cv::Mat()));

    // linearity should be bad
    EXPECT_GT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);

    // final depth estimation is pushed far away
    EXPECT_LT(inverseDepth._coordinates.get_inverse_depth(), parameters::detection::inverseDepthBaseline_m);
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
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix66::Zero(), cv::Mat());

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

    Eigen::Matrix<double, 3, 6> toWorldJacobian;
    auto worldProj = inverseDepth._coordinates.to_world_coordinates(toWorldJacobian);
    auto worldCov =
            tracking::PointInverseDepth::compute_cartesian_covariance(inverseDepth._covariance, toWorldJacobian);

    // check world covariance projection
    // high on X, fairly ok in Y Z
    EXPECT_GT(worldCov(0, 0), worldCov(1, 1));
    EXPECT_GT(worldCov(0, 0), worldCov(2, 2));
    EXPECT_LT(worldCov(1, 1), 10000);
    EXPECT_LT(worldCov(2, 2), 10000);
    // all other are zero
    EXPECT_NEAR(worldCov(0, 1), 0.0, 1e-4);
    EXPECT_NEAR(worldCov(1, 0), 0.0, 1e-4);
    EXPECT_NEAR(worldCov(0, 2), 0.0, 1e-4);
    EXPECT_NEAR(worldCov(2, 0), 0.0, 1e-4);
    EXPECT_NEAR(worldCov(1, 2), 0.0, 1e-4);
    EXPECT_NEAR(worldCov(2, 1), 0.0, 1e-4);

    // check that the projection is indeed at the baseline
    EXPECT_NEAR(worldProj.x(), 1.0 / parameters::detection::inverseDepthBaseline_m, 1e-5);
    EXPECT_NEAR(worldProj.y(), 0.0, 1e-5);
    EXPECT_NEAR(worldProj.z(), 0.0, 1e-5);

    /**
     ** add a new measurment at 90° from the position, further on the trajectory
     */
    const double observedZ = 10.0;
    const CameraToWorldMatrix& c2wSide90 = utils::compute_camera_to_world_transform(
            utils::get_quaternion_from_euler_angles(EulerAngles(0.0, -90 * EulerToRadian, 0.0)),
            vector3(1.0, 0.0, observedZ));

    utils::Segment<2> screenSegment;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2wSide90), screenSegment));
    // this is a line, should always be a line throught center of the "screen"
    EXPECT_NEAR(screenSegment.get_start_point().y(), observation.y(), 1);
    EXPECT_NEAR(screenSegment.get_end_point().y(), observation.y(), 1);
    // start x is outside the left of the image, end point x is outside the right side
    EXPECT_LT(screenSegment.get_start_point().x(), 0.0);
    EXPECT_GT(screenSegment.get_end_point().x(), Parameters::get_camera_1_image_size().x());

    EXPECT_GT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);

    // fuse the two points (multiple observations, process is slow)
    for (uint i = 0; i < 5; i++)
    {
        assert_inverse_point_back_proj(c2wSide90, observation);

        EXPECT_TRUE(inverseDepth.track_2D(
                observation, observation.get_covariance(), c2wSide90, matrix66::Identity(), cv::Mat()));

        EXPECT_TRUE(
                inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2wSide90), screenSegment));

        // this is a line, should always be a line throught center of the "screen"
        EXPECT_NEAR(screenSegment.get_start_point().y(), observation.y(), 1);
        EXPECT_NEAR(screenSegment.get_end_point().y(), observation.y(), 1);

        // this projection gives back the depth in screen space, check that it is close to expected
        const auto finalPoint = inverseDepth._coordinates.to_world_coordinates();
        ScreenCoordinate sc;
        EXPECT_TRUE(finalPoint.to_screen_coordinates(utils::compute_world_to_camera_transform(c2wSide90), sc));

#ifdef DEBUG_TESTS
        std::cout << std::endl << inverseDepth._coordinates.get_vector().transpose() << std::endl;
        std::cout << "  after merge " << finalPoint.transpose() << std::endl;
        std::cout << "  after merge proj " << sc.transpose() << std::endl;
        std::cout << "  segment x [" << screenSegment.get_start_point().x() << " ; "
                  << screenSegment.get_end_point().x() << "]" << std::endl;
#endif

        // x has too much uncertainty during opti process, do not check it
        EXPECT_NEAR(sc.y(), observation.y(), 1);
        EXPECT_NEAR(sc.z(), 1.0, 0.01);
    }

    // check that the final projection line is close around the target
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
    EXPECT_NEAR(inverseDepth._coordinates.get_inverse_depth(), 1.0 / observedZ, 0.0001);

    EXPECT_NEAR(finalPoint.x(), observedZ, 0.01);
    // 1cm tolerance
    EXPECT_NEAR(finalPoint.y(), 0, 0.01);
    EXPECT_NEAR(finalPoint.z(), 0, 0.01);

    // linearity should be pretty good
    EXPECT_LT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);
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
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix66::Zero(), cv::Mat());

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

    Eigen::Matrix<double, 3, 6> toWorldJacobian;
    auto worldProj = inverseDepth._coordinates.to_world_coordinates(toWorldJacobian);
    auto worldCov =
            tracking::PointInverseDepth::compute_cartesian_covariance(inverseDepth._covariance, toWorldJacobian);

    // check world covariance projection
    // high on X, fairly ok in Y Z
    EXPECT_GT(worldCov(0, 0), worldCov(1, 1));
    EXPECT_GT(worldCov(0, 0), worldCov(2, 2));
    EXPECT_LT(worldCov(1, 1), 10000);
    EXPECT_LT(worldCov(2, 2), 10000);
    // all other are zero
    EXPECT_NEAR(worldCov(0, 1), 0.0, 1e-4);
    EXPECT_NEAR(worldCov(1, 0), 0.0, 1e-4);
    EXPECT_NEAR(worldCov(0, 2), 0.0, 1e-4);
    EXPECT_NEAR(worldCov(2, 0), 0.0, 1e-4);
    EXPECT_NEAR(worldCov(1, 2), 0.0, 1e-4);
    EXPECT_NEAR(worldCov(2, 1), 0.0, 1e-4);

    // check that the projection is indeed at the baseline
    EXPECT_NEAR(worldProj.x(), 1.0 / parameters::detection::inverseDepthBaseline_m, 1e-5);
    EXPECT_NEAR(worldProj.y(), 0.0, 1e-5);
    EXPECT_NEAR(worldProj.z(), 0.0, 1e-5);

    /**
     ** add a new measurment at 90° from the position, further on the trajectory
     */
    const double observedZ = 10.0;
    const CameraToWorldMatrix& c2wSide90 = utils::compute_camera_to_world_transform(
            utils::get_quaternion_from_euler_angles(EulerAngles(0.0, 90 * EulerToRadian, 0.0)),
            vector3(-1.0, 0.0, observedZ));

    utils::Segment<2> screenSegment;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2wSide90), screenSegment));
    // this is a line, should always be a line throught center of the "screen"
    EXPECT_NEAR(screenSegment.get_start_point().y(), observation.y(), 1);
    EXPECT_NEAR(screenSegment.get_end_point().y(), observation.y(), 1);
    // start x is outside the left of the image, end point x is outside the right side
    EXPECT_LT(screenSegment.get_end_point().x(), 0.0);
    EXPECT_GT(screenSegment.get_start_point().x(), Parameters::get_camera_1_image_size().x());

    EXPECT_GT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);

    // fuse the two points (multiple observations, process is slow)
    for (uint i = 0; i < 5; i++)
    {
        assert_inverse_point_back_proj(c2wSide90, observation);

        EXPECT_TRUE(inverseDepth.track_2D(
                observation, observation.get_covariance(), c2wSide90, matrix66::Zero(), cv::Mat()));

        EXPECT_TRUE(
                inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2wSide90), screenSegment));

        // this is a line, should always be a line throught center of the "screen"
        EXPECT_NEAR(screenSegment.get_start_point().y(), observation.y(), 1);
        EXPECT_NEAR(screenSegment.get_end_point().y(), observation.y(), 1);

        // this projection gives back the depth in screen space, check that it is close to expected
        const auto finalPoint = inverseDepth._coordinates.to_world_coordinates();
        ScreenCoordinate sc;
        EXPECT_TRUE(finalPoint.to_screen_coordinates(utils::compute_world_to_camera_transform(c2wSide90), sc));

#ifdef DEBUG_TESTS
        std::cout << std::endl << inverseDepth._coordinates.get_vector().transpose() << std::endl;
        std::cout << "  after merge " << finalPoint.transpose() << std::endl;
        std::cout << "  after merge proj " << sc.transpose() << std::endl;
        std::cout << "  segment x [" << screenSegment.get_start_point().x() << " ; "
                  << screenSegment.get_end_point().x() << "]" << std::endl;
#endif

        // x has too much uncertainty during opti process, do not check it
        EXPECT_NEAR(sc.y(), observation.y(), 1);
        EXPECT_NEAR(sc.z(), 1.0, 0.01);
    }

    // check that the final projection line is close around the target
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
    EXPECT_NEAR(inverseDepth._coordinates.get_inverse_depth(), 1.0 / observedZ, 0.0001);

    EXPECT_NEAR(finalPoint.x(), observedZ, 0.01);
    // 1cm tolerance
    EXPECT_NEAR(finalPoint.y(), 0, 0.01);
    EXPECT_NEAR(finalPoint.z(), 0, 0.01);

    // linearity should be pretty good
    EXPECT_LT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);
}

TEST(InverseDepthPointFusion, fusePointObservationXAxis)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const WorldCoordinate pointToTrack(1.0, 0.2, 0.5);

    // observe the center of the camera
    CameraToWorldMatrix c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());
    WorldToCameraMatrix w2c = utils::compute_world_to_camera_transform(c2w);
    ScreenCoordinate2D observation;
    ASSERT_TRUE(pointToTrack.to_screen_coordinates(w2c, observation));

    // convert to inverse
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix66::Zero(), cv::Mat());

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
     ** add a new measurment at another point in space
     */
    double lastThreshold = inverseDepth.compute_linearity_score(c2w);
    EXPECT_GT(lastThreshold, linearityThreshold);

    for (double i = 0.1; i < 0.5; i += 0.02)
    {
        c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(i, 0.0, 0.0));
        w2c = utils::compute_world_to_camera_transform(c2w);
        // make new observation
        ASSERT_TRUE(pointToTrack.to_screen_coordinates(w2c, observation));

        assert_inverse_point_back_proj(c2w, observation);

        matrix22 cov = vector2::Constant(SQR(2.0)).asDiagonal();
        EXPECT_TRUE(inverseDepth.track_2D(observation, cov, c2w, matrix66::Zero(), cv::Mat()));

        const double newThreshold = inverseDepth.compute_linearity_score(c2w);
        EXPECT_LT(newThreshold, lastThreshold); // still less than
        lastThreshold = newThreshold;

        // check that the projection line is close around the target
        utils::Segment<2> screenSegment;
        EXPECT_TRUE(inverseDepth.to_screen_coordinates(w2c, screenSegment));
        EXPECT_LE(screenSegment.get_start_point().x() - 1e-3, observation.x());
        EXPECT_GE(screenSegment.get_end_point().x() + 1e-3, observation.x());
        EXPECT_LE(screenSegment.get_start_point().y() - 1e-3, observation.y());
        EXPECT_GE(screenSegment.get_end_point().y() + 1e-3, observation.y());
    }

    utils::Segment<2> screenSegment;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(w2c, screenSegment));

    // this projection gives back the depth in screen space, check that it is close to expected
    auto finalPoint = inverseDepth._coordinates.to_world_coordinates();

    // check that the final projection line is close around the target
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(w2c, screenSegment));
    EXPECT_LE(screenSegment.get_start_point().x() - 1e-3, observation.x());
    EXPECT_GE(screenSegment.get_end_point().x() + 1e-3, observation.x());
    EXPECT_LE(screenSegment.get_start_point().y() - 1e-3, observation.y());
    EXPECT_GE(screenSegment.get_end_point().y() + 1e-3, observation.y());

    finalPoint = inverseDepth._coordinates.to_world_coordinates();
    const auto finalPointCovariance = tracking::PointInverseDepth::compute_cartesian_covariance(
            inverseDepth._coordinates, inverseDepth._covariance);
    EXPECT_TRUE(utils::is_covariance_valid(finalPointCovariance));

    // final pose is triangulated (1cm tolerance)
    EXPECT_NEAR(finalPoint.x(), pointToTrack.x(), 0.01);
    EXPECT_NEAR(finalPoint.y(), pointToTrack.y(), 0.01);
    EXPECT_NEAR(finalPoint.z(), pointToTrack.z(), 0.01);

    // linearity should be pretty good
    EXPECT_LT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);
}

TEST(InverseDepthPointFusion, fusePointObservationYAxis)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const WorldCoordinate pointToTrack(1.0, 0.2, 0.5);

    // observe the center of the camera
    CameraToWorldMatrix c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());
    WorldToCameraMatrix w2c = utils::compute_world_to_camera_transform(c2w);
    ScreenCoordinate2D observation;
    ASSERT_TRUE(pointToTrack.to_screen_coordinates(w2c, observation));

    // convert to inverse
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix66::Zero(), cv::Mat());

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
     ** add a new measurment at another point in space
     */
    double lastThreshold = inverseDepth.compute_linearity_score(c2w);
    EXPECT_GT(lastThreshold, linearityThreshold);

    for (double i = 0.1; i < 0.5; i += 0.02)
    {
        c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(0.0, -i, 0.0));
        w2c = utils::compute_world_to_camera_transform(c2w);
        // make new observation
        ASSERT_TRUE(pointToTrack.to_screen_coordinates(w2c, observation));

        assert_inverse_point_back_proj(c2w, observation);

        matrix22 cov = vector2::Constant(SQR(2.0)).asDiagonal();
        EXPECT_TRUE(inverseDepth.track_2D(observation, cov, c2w, matrix66::Zero(), cv::Mat()));

        const double newThreshold = inverseDepth.compute_linearity_score(c2w);
        EXPECT_LT(newThreshold, lastThreshold); // still less than
        lastThreshold = newThreshold;

        utils::Segment<2> screenSegment;
        EXPECT_TRUE(inverseDepth.to_screen_coordinates(w2c, screenSegment));

        EXPECT_LE(screenSegment.get_end_point().x() - 1e-3, observation.x());
        EXPECT_GE(screenSegment.get_start_point().x() + 1e-3, observation.x());
        EXPECT_LE(screenSegment.get_end_point().y() - 1e-3, observation.y());
        EXPECT_GE(screenSegment.get_start_point().y() + 1e-3, observation.y());
    }

    utils::Segment<2> screenSegment;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenSegment));

    // this projection gives back the depth in screen space, check that it is close to expected
    auto finalPoint = inverseDepth._coordinates.to_world_coordinates();

    // check that the final projection line is close around the target
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenSegment));
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(w2c, screenSegment));
    EXPECT_LE(screenSegment.get_end_point().x() - 1e-3, observation.x());
    EXPECT_GE(screenSegment.get_start_point().x() + 1e-3, observation.x());
    EXPECT_LE(screenSegment.get_end_point().y() - 1e-3, observation.y());
    EXPECT_GE(screenSegment.get_start_point().y() + 1e-3, observation.y());

    finalPoint = inverseDepth._coordinates.to_world_coordinates();
    const auto finalPointCovariance = tracking::PointInverseDepth::compute_cartesian_covariance(
            inverseDepth._coordinates, inverseDepth._covariance);
    EXPECT_TRUE(utils::is_covariance_valid(finalPointCovariance));

    // final pose is triangulated (1cm tolerance)
    EXPECT_NEAR(finalPoint.x(), pointToTrack.x(), 0.01);
    EXPECT_NEAR(finalPoint.y(), pointToTrack.y(), 0.01);
    EXPECT_NEAR(finalPoint.z(), pointToTrack.z(), 0.01);

    // linearity should be pretty good
    EXPECT_LT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);
}

TEST(InverseDepthPointFusion, fusePointObservationZAxis)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const WorldCoordinate pointToTrack(1.0, 0.2, 0.5);

    // observe the center of the camera
    CameraToWorldMatrix c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());
    WorldToCameraMatrix w2c = utils::compute_world_to_camera_transform(c2w);
    ScreenCoordinate2D observation;
    ASSERT_TRUE(pointToTrack.to_screen_coordinates(w2c, observation));

    // convert to inverse
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix66::Zero(), cv::Mat());

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
     ** add a new measurment at another point in space
     */
    double lastThreshold = inverseDepth.compute_linearity_score(c2w);
    EXPECT_GT(lastThreshold, linearityThreshold);

    for (double i = 0.1; i < 0.5; i += 0.02)
    {
        c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(0.0, 0.0, i));
        w2c = utils::compute_world_to_camera_transform(c2w);
        // make new observation
        ASSERT_TRUE(pointToTrack.to_screen_coordinates(w2c, observation));

        assert_inverse_point_back_proj(c2w, observation);

        matrix22 cov = vector2::Constant(SQR(2.0)).asDiagonal();
        EXPECT_TRUE(inverseDepth.track_2D(observation, cov, c2w, matrix66::Zero(), cv::Mat()));

        const double newThreshold = inverseDepth.compute_linearity_score(c2w);
        EXPECT_LT(newThreshold, lastThreshold); // still less than
        lastThreshold = newThreshold;

        utils::Segment<2> screenSegment;
        EXPECT_TRUE(inverseDepth.to_screen_coordinates(w2c, screenSegment));
        EXPECT_LE(screenSegment.get_start_point().x() - 1e-3, observation.x());
        EXPECT_GE(screenSegment.get_end_point().x() + 1e-3, observation.x());
        EXPECT_LE(screenSegment.get_start_point().y() - 1e-3, observation.y());
        EXPECT_GE(screenSegment.get_end_point().y() + 1e-3, observation.y());
    }

    utils::Segment<2> screenSegment;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenSegment));

    // this projection gives back the depth in screen space, check that it is close to expected
    auto finalPoint = inverseDepth._coordinates.to_world_coordinates();

    // check that the final projection line is close around the target
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenSegment));
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(w2c, screenSegment));
    EXPECT_LE(screenSegment.get_start_point().x() - 1e-3, observation.x());
    EXPECT_GE(screenSegment.get_end_point().x() + 1e-3, observation.x());
    EXPECT_LE(screenSegment.get_start_point().y() - 1e-3, observation.y());
    EXPECT_GE(screenSegment.get_end_point().y() + 1e-3, observation.y());

    finalPoint = inverseDepth._coordinates.to_world_coordinates();
    const auto finalPointCovariance = tracking::PointInverseDepth::compute_cartesian_covariance(
            inverseDepth._coordinates, inverseDepth._covariance);
    EXPECT_TRUE(utils::is_covariance_valid(finalPointCovariance));

    // final pose is triangulated (1cm tolerance)
    EXPECT_NEAR(finalPoint.x(), pointToTrack.x(), 0.01);
    EXPECT_NEAR(finalPoint.y(), pointToTrack.y(), 0.01);
    EXPECT_NEAR(finalPoint.z(), pointToTrack.z(), 0.01);

    // linearity should be pretty good
    EXPECT_LT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);
}

TEST(InverseDepthPointFusion, fusePointObservationZAxisFarAway)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const WorldCoordinate pointToTrack(100.0, 0.2, 0.5);

    // observe the center of the camera
    CameraToWorldMatrix c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());
    WorldToCameraMatrix w2c = utils::compute_world_to_camera_transform(c2w);
    ScreenCoordinate2D observation;
    ASSERT_TRUE(pointToTrack.to_screen_coordinates(w2c, observation));

    // convert to inverse
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix66::Zero(), cv::Mat());

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
     ** add a new measurment at another point in space
     */
    double lastThreshold = inverseDepth.compute_linearity_score(c2w);
    EXPECT_GT(lastThreshold, linearityThreshold);

    for (double i = 0.1; i < 0.5; i += 0.02)
    {
        c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(0.0, 0.0, i));
        w2c = utils::compute_world_to_camera_transform(c2w);
        // make new observation
        ASSERT_TRUE(pointToTrack.to_screen_coordinates(w2c, observation));

        assert_inverse_point_back_proj(c2w, observation);

        matrix22 cov = vector2::Constant(SQR(2.0)).asDiagonal();
        EXPECT_TRUE(inverseDepth.track_2D(observation, cov, c2w, matrix66::Zero(), cv::Mat()));
    }

    // linearity should be bad, point is not triangulated
    EXPECT_GT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);
    // in fact, unlinearity got bigger
    EXPECT_GT(inverseDepth.compute_linearity_score(c2w), lastThreshold);
}

/**
 * Inverse depth point fusion with 3D coordinates
 */

TEST(InverseDepthPointFusion3d, fusePointObservationXAxis)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const WorldCoordinate pointToTrack(1.0, 0.2, 0.5);

    // observe the center of the camera
    CameraToWorldMatrix c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());
    WorldToCameraMatrix w2c = utils::compute_world_to_camera_transform(c2w);
    ScreenCoordinate observation;
    ScreenCoordinate2D observation2d;
    ASSERT_TRUE(pointToTrack.to_screen_coordinates(w2c, observation2d));

    // convert to inverse
    tracking::PointInverseDepth inverseDepth(observation2d, c2w, matrix66::Zero(), cv::Mat());

    // check projection/backprojection
    assert_inverse_point_back_proj(c2w, observation2d);
    assert_inverse_point_back_proj(c2w, observation.get_2D());

    // check that the projected segment is in fact a point
    utils::Segment<2> originalSegment;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), originalSegment));
    EXPECT_NEAR(originalSegment.get_start_point().x(), observation2d.x(), 1);
    EXPECT_NEAR(originalSegment.get_start_point().y(), observation2d.y(), 1);
    EXPECT_NEAR(originalSegment.get_end_point().x(), observation2d.x(), 1);
    EXPECT_NEAR(originalSegment.get_end_point().y(), observation2d.y(), 1);

    EXPECT_NEAR(originalSegment.get_end_point().x(), originalSegment.get_start_point().x(), 1);
    EXPECT_NEAR(originalSegment.get_end_point().y(), originalSegment.get_start_point().y(), 1);

    /**
     ** add a new measurment at another point in space
     */
    double lastThreshold = inverseDepth.compute_linearity_score(c2w);
    EXPECT_GT(lastThreshold, linearityThreshold);

    for (double i = 0.1; i < 0.2; i += 0.02)
    {
        c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(i, 0, 0.0));
        w2c = utils::compute_world_to_camera_transform(c2w);

        // make new observation
        ASSERT_TRUE(pointToTrack.to_screen_coordinates(w2c, observation));

        assert_inverse_point_back_proj(c2w, observation.get_2D());

        matrix33 covariance = vector3(1, 1, SQR(0.1)).asDiagonal();
        EXPECT_TRUE(inverseDepth.track_3D(observation, covariance, c2w, matrix66::Zero(), cv::Mat()));

        // check that the projection line is close around the target
        utils::Segment<2> screenSegment;
        EXPECT_TRUE(inverseDepth.to_screen_coordinates(w2c, screenSegment));
        EXPECT_LE(screenSegment.get_start_point().x() - 1e-3, observation.x());
        EXPECT_GE(screenSegment.get_end_point().x() + 1e-3, observation.x());
        EXPECT_LE(screenSegment.get_start_point().y() - 1e-3, observation.y());
        EXPECT_GE(screenSegment.get_end_point().y() + 1e-3, observation.y());
    }

    // this projection gives back the depth in screen space, check that it is close to expected
    auto finalPoint = inverseDepth._coordinates.to_world_coordinates();

    finalPoint = inverseDepth._coordinates.to_world_coordinates();
    const auto finalPointCovariance = tracking::PointInverseDepth::compute_cartesian_covariance(
            inverseDepth._coordinates, inverseDepth._covariance);
    EXPECT_TRUE(utils::is_covariance_valid(finalPointCovariance));

    // final pose is triangulated (1cm tolerance)
    EXPECT_NEAR(finalPoint.x(), pointToTrack.x(), 0.01);
    EXPECT_NEAR(finalPoint.y(), pointToTrack.y(), 0.01);
    EXPECT_NEAR(finalPoint.z(), pointToTrack.z(), 0.01);

    // linearity should be pretty good
    EXPECT_LT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);
}

TEST(InverseDepthPointFusion3d, fusePointObservationYAxis)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const WorldCoordinate pointToTrack(1.0, 0.2, 0.5);

    // observe the center of the camera
    CameraToWorldMatrix c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());
    WorldToCameraMatrix w2c = utils::compute_world_to_camera_transform(c2w);
    ScreenCoordinate observation;
    ScreenCoordinate2D observation2d;
    ASSERT_TRUE(pointToTrack.to_screen_coordinates(w2c, observation2d));

    // convert to inverse
    tracking::PointInverseDepth inverseDepth(observation2d, c2w, matrix66::Zero(), cv::Mat());

    // check projection/backprojection
    assert_inverse_point_back_proj(c2w, observation2d);
    assert_inverse_point_back_proj(c2w, observation.get_2D());

    // check that the projected segment is in fact a point
    utils::Segment<2> originalSegment;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), originalSegment));
    EXPECT_NEAR(originalSegment.get_start_point().x(), observation2d.x(), 1);
    EXPECT_NEAR(originalSegment.get_start_point().y(), observation2d.y(), 1);
    EXPECT_NEAR(originalSegment.get_end_point().x(), observation2d.x(), 1);
    EXPECT_NEAR(originalSegment.get_end_point().y(), observation2d.y(), 1);

    EXPECT_NEAR(originalSegment.get_end_point().x(), originalSegment.get_start_point().x(), 1);
    EXPECT_NEAR(originalSegment.get_end_point().y(), originalSegment.get_start_point().y(), 1);

    /**
     ** add a new measurment at another point in space
     */
    double lastThreshold = inverseDepth.compute_linearity_score(c2w);
    EXPECT_GT(lastThreshold, linearityThreshold);

    for (double i = 0.1; i < 0.2; i += 0.02)
    {
        c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(0, i, 0.0));
        w2c = utils::compute_world_to_camera_transform(c2w);

        // make new observation
        ASSERT_TRUE(pointToTrack.to_screen_coordinates(w2c, observation));

        assert_inverse_point_back_proj(c2w, observation.get_2D());

        matrix33 covariance = vector3(1, 1, SQR(0.1)).asDiagonal();
        EXPECT_TRUE(inverseDepth.track_3D(observation, covariance, c2w, matrix66::Zero(), cv::Mat()));

        // check that the projection line is close around the target
        utils::Segment<2> screenSegment;
        EXPECT_TRUE(inverseDepth.to_screen_coordinates(w2c, screenSegment));
        EXPECT_LE(screenSegment.get_start_point().x() - 1e-3, observation.x());
        EXPECT_GE(screenSegment.get_end_point().x() + 1e-3, observation.x());
        EXPECT_LE(screenSegment.get_start_point().y() - 1e-3, observation.y());
        EXPECT_GE(screenSegment.get_end_point().y() + 1e-3, observation.y());
    }

    // this projection gives back the depth in screen space, check that it is close to expected
    auto finalPoint = inverseDepth._coordinates.to_world_coordinates();

    finalPoint = inverseDepth._coordinates.to_world_coordinates();
    const auto finalPointCovariance = tracking::PointInverseDepth::compute_cartesian_covariance(
            inverseDepth._coordinates, inverseDepth._covariance);
    EXPECT_TRUE(utils::is_covariance_valid(finalPointCovariance));

    // final pose is triangulated (1cm tolerance)
    EXPECT_NEAR(finalPoint.x(), pointToTrack.x(), 0.01);
    EXPECT_NEAR(finalPoint.y(), pointToTrack.y(), 0.01);
    EXPECT_NEAR(finalPoint.z(), pointToTrack.z(), 0.01);

    // linearity should be pretty good
    EXPECT_LT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);
}

TEST(InverseDepthPointFusion3d, fusePointObservationZAxis)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const WorldCoordinate pointToTrack(1.0, 0.2, 0.5);

    // observe the center of the camera
    CameraToWorldMatrix c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());
    WorldToCameraMatrix w2c = utils::compute_world_to_camera_transform(c2w);
    ScreenCoordinate observation;
    ScreenCoordinate2D observation2d;
    ASSERT_TRUE(pointToTrack.to_screen_coordinates(w2c, observation2d));

    // convert to inverse
    tracking::PointInverseDepth inverseDepth(observation2d, c2w, matrix66::Zero(), cv::Mat());

    // check projection/backprojection
    assert_inverse_point_back_proj(c2w, observation2d);
    assert_inverse_point_back_proj(c2w, observation.get_2D());

    // check that the projected segment is in fact a point
    utils::Segment<2> originalSegment;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), originalSegment));
    EXPECT_NEAR(originalSegment.get_start_point().x(), observation2d.x(), 1);
    EXPECT_NEAR(originalSegment.get_start_point().y(), observation2d.y(), 1);
    EXPECT_NEAR(originalSegment.get_end_point().x(), observation2d.x(), 1);
    EXPECT_NEAR(originalSegment.get_end_point().y(), observation2d.y(), 1);

    EXPECT_NEAR(originalSegment.get_end_point().x(), originalSegment.get_start_point().x(), 1);
    EXPECT_NEAR(originalSegment.get_end_point().y(), originalSegment.get_start_point().y(), 1);

    /**
     ** add a new measurment at another point in space
     */
    double lastThreshold = inverseDepth.compute_linearity_score(c2w);
    EXPECT_GT(lastThreshold, linearityThreshold);

    for (double i = 0.1; i < 0.2; i += 0.02)
    {
        c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(0, 0, i));
        w2c = utils::compute_world_to_camera_transform(c2w);

        // make new observation
        ASSERT_TRUE(pointToTrack.to_screen_coordinates(w2c, observation));

        assert_inverse_point_back_proj(c2w, observation.get_2D());

        matrix33 covariance = vector3(1, 1, SQR(0.1)).asDiagonal();
        EXPECT_TRUE(inverseDepth.track_3D(observation, covariance, c2w, matrix66::Zero(), cv::Mat()));

        // check that the projection line is close around the target
        utils::Segment<2> screenSegment;
        EXPECT_TRUE(inverseDepth.to_screen_coordinates(w2c, screenSegment));
        EXPECT_LE(screenSegment.get_start_point().x() - 1e-3, observation.x());
        EXPECT_GE(screenSegment.get_end_point().x() + 1e-3, observation.x());
        EXPECT_LE(screenSegment.get_start_point().y() - 1e-3, observation.y());
        EXPECT_GE(screenSegment.get_end_point().y() + 1e-3, observation.y());
    }

    // this projection gives back the depth in screen space, check that it is close to expected
    auto finalPoint = inverseDepth._coordinates.to_world_coordinates();

    finalPoint = inverseDepth._coordinates.to_world_coordinates();
    const auto finalPointCovariance = tracking::PointInverseDepth::compute_cartesian_covariance(
            inverseDepth._coordinates, inverseDepth._covariance);
    EXPECT_TRUE(utils::is_covariance_valid(finalPointCovariance));

    // final pose is triangulated (1cm tolerance)
    EXPECT_NEAR(finalPoint.x(), pointToTrack.x(), 0.01);
    EXPECT_NEAR(finalPoint.y(), pointToTrack.y(), 0.01);
    EXPECT_NEAR(finalPoint.z(), pointToTrack.z(), 0.01);

    // linearity should be pretty good
    EXPECT_LT(inverseDepth.compute_linearity_score(c2w), linearityThreshold);
}

} // namespace rgbd_slam::tracking
