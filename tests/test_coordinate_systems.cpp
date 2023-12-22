#include "angle_utils.hpp"
#include "covariances.hpp"
#include "distance_utils.hpp"
#include "parameters.hpp"
#include "point_with_tracking.hpp"
#include "types.hpp"
#include "utils/camera_transformation.hpp"
#include "utils/coordinates/point_coordinates.hpp"
#include "utils/coordinates/plane_coordinates.hpp"
#include <gtest/gtest.h>
#include <iostream>

namespace rgbd_slam::utils {

void estimate_point_error(const vector3& pointA, const vector3& pointB)
{
    EXPECT_NEAR(pointA.x(), pointB.x(), 0.001);
    EXPECT_NEAR(pointA.y(), pointB.y(), 0.001);
    EXPECT_NEAR(pointA.z(), pointB.z(), 0.001);
}

TEST(CoordinateSystemChangeTests, CameraToWorldAtOrigin)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(0, 0, 0));

    const matrix44& tr = utils::get_transformation_matrix(
            vector3(1, 0, 0), vector3(0, 1, 0), vector3::Zero(), vector3(1, 0, 0), vector3(0, 1, 0), vector3::Zero());

    EXPECT_TRUE(cameraToWorld.isApprox(tr));
}

TEST(CoordinateSystemChangeTests, CameraToWorldFarFromOrigin)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(-100, 100, 200));

    const matrix44& tr = utils::get_transformation_matrix(vector3(1, 0, 0),
                                                          vector3(0, 1, 0),
                                                          vector3::Zero(),
                                                          vector3(1, 0, 0),
                                                          vector3(0, 1, 0),
                                                          vector3(-100, 100, 200));

    EXPECT_TRUE(cameraToWorld.isApprox(tr));
}

TEST(CoordinateSystemChangeTests, CameraToWorldAtOriginWithRotation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform_no_correction(quaternion(0.0, 1.0, 0.0, 0.0), vector3(0, 0, 0));

    const matrix44& tr = utils::get_transformation_matrix(
            vector3(1, 0, 0), vector3(0, 1, 0), vector3::Zero(), vector3(1, 0, 0), vector3(0, -1, 0), vector3::Zero());

    EXPECT_TRUE(cameraToWorld.isApprox(tr));
}

TEST(CoordinateSystemChangeTests, CameraToWorldAtOriginWithRotation2)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform_no_correction(quaternion(0.0, 0.0, 1.0, 0.0), vector3(0, 0, 0));

    const matrix44& tr = utils::get_transformation_matrix(
            vector3(1, 0, 0), vector3(0, 1, 0), vector3::Zero(), vector3(-1, 0, 0), vector3(0, 1, 0), vector3::Zero());

    EXPECT_TRUE(cameraToWorld.isApprox(tr));
}

TEST(CoordinateSystemChangeTests, CameraToWorldAtOriginWithRotation3)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform_no_correction(quaternion(0.0, 0, 0, 1.0), vector3(0, 0, 0));

    const matrix44& tr = utils::get_transformation_matrix(
            vector3(1, 0, 0), vector3(0, 1, 0), vector3::Zero(), vector3(-1, 0, 0), vector3(0, -1, 0), vector3::Zero());

    EXPECT_TRUE(cameraToWorld.isApprox(tr));
}

TEST(CoordinateSystemChangeTests, CameraToWorldAtOriginWithRotationCombined)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform_no_correction(quaternion(0.5, 0.5, 0.5, 0.5), vector3(0, 0, 0));

    const matrix44& tr = utils::get_transformation_matrix(
            vector3(1, 0, 0), vector3(0, 1, 0), vector3::Zero(), vector3(0, 1, 0), vector3(0, 0, 1), vector3::Zero());

    EXPECT_TRUE(cameraToWorld.isApprox(tr));
}

TEST(CoordinateSystemChangeTests, CameraToWorldFarFromOriginWithRotation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform_no_correction(quaternion(0.0, 1.0, 0.0, 0.0), vector3(-100, 100, 200));

    const matrix44& tr = utils::get_transformation_matrix(vector3(1, 0, 0),
                                                          vector3(0, 1, 0),
                                                          vector3::Zero(),
                                                          vector3(1, 0, 0),
                                                          vector3(0, -1, 0),
                                                          vector3(-100, 100, 200));
    EXPECT_TRUE(cameraToWorld.isApprox(tr));
}

TEST(CoordinateSystemChangeTests, CameraToWorldFarFromOriginSameWithRotation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform_no_correction(quaternion(0.0, 1.0, 0.0, 0.0), vector3(0, 0, 0));

    const matrix44& tr = utils::get_transformation_matrix(vector3(1, 0, 0),
                                                          vector3(0, 1, 0),
                                                          vector3(-100, 100, 200),
                                                          vector3(1, 0, 0),
                                                          vector3(0, -1, 0),
                                                          vector3(-100, 100, 200));
    EXPECT_TRUE(cameraToWorld.isApprox(tr));
}

TEST(PointCoordinateSystemTests, ScreenToCameraToScreen)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const double xRange = 640.0;
    const double yRange = 480.0;
    const double zRange = 50.0;
    const double xIncrement = 7.5;
    const double yIncrement = 5.5;
    const double zIncrement = 0.5;
    for (double x = 0; x < xRange; x += xIncrement)
    {
        for (double y = 0; y < yRange; y += yIncrement)
        {
            for (double z = zIncrement; z < zRange; z += zIncrement)
            {
                const ScreenCoordinate originalScreenCoordinates(x, y, z);
                const CameraCoordinate cameraCoordinates = originalScreenCoordinates.to_camera_coordinates();
                ScreenCoordinate newScreenCoordinates;
                if (cameraCoordinates.to_screen_coordinates(newScreenCoordinates))
                {
                    estimate_point_error(originalScreenCoordinates, newScreenCoordinates);
                }
                else
                {
                    FAIL();
                }
            }

            for (double z = -zRange; z < -zIncrement; z += zIncrement)
            {
                const ScreenCoordinate originalScreenCoordinates(x, y, z);
                const CameraCoordinate cameraCoordinates = originalScreenCoordinates.to_camera_coordinates();
                ScreenCoordinate newScreenCoordinates;
                if (cameraCoordinates.to_screen_coordinates(newScreenCoordinates))
                {
                    estimate_point_error(originalScreenCoordinates, newScreenCoordinates);
                }
                else
                {
                    FAIL();
                }
            }
        }
    }
}

void test_point_set_screen_to_world_to_screen(const CameraToWorldMatrix& cameraToWorld)
{
    const WorldToCameraMatrix worldToCamera = compute_world_to_camera_transform(cameraToWorld);

    const double xRange = 640.0;
    const double yRange = 480.0;
    const double zRange = 50.0;
    const double xIncrement = 7.5;
    const double yIncrement = 5.5;
    const double zIncrement = 0.5;
    for (double x = 0; x < xRange; x += xIncrement)
    {
        for (double y = 0; y < yRange; y += yIncrement)
        {
            for (double z = zIncrement; z < zRange; z += zIncrement)
            {
                const ScreenCoordinate originalScreenCoordinates(x, y, z);
                const WorldCoordinate worldCoordinates = originalScreenCoordinates.to_world_coordinates(cameraToWorld);
                ScreenCoordinate newScreenCoordinates;
                if (worldCoordinates.to_screen_coordinates(worldToCamera, newScreenCoordinates))
                {
                    estimate_point_error(originalScreenCoordinates, newScreenCoordinates);
                }
                else
                {
                    FAIL();
                }
            }
        }
    }
}

TEST(PointCoordinateSystemTests, ScreenToWorldToScreenAtOrigin)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(0, 0, 0));
    test_point_set_screen_to_world_to_screen(cameraToWorld);
}

TEST(PointCoordinateSystemTests, ScreenToWorldToScreenFarFromOrigin)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(-100, 1000, 100));
    test_point_set_screen_to_world_to_screen(cameraToWorld);
}

TEST(PointCoordinateSystemTests, ScreenToWorldToScreenRotation1)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform_no_correction(quaternion(0.3, 0.2, 0.1, 0.4), vector3(0, 0, 0));
    test_point_set_screen_to_world_to_screen(cameraToWorld);
}

TEST(PointCoordinateSystemTests, ScreenToWorldToScreenRotation2)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform_no_correction(quaternion(0.6, 0.1, 0.2, 0.1), vector3(0, 0, 0));
    test_point_set_screen_to_world_to_screen(cameraToWorld);
}

TEST(PointCoordinateSystemTests, ScreenToWorldToScreenRotation3)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform_no_correction(quaternion(0.6, 0.1, 0.2, 0.1), vector3(100, -100, -100));
    test_point_set_screen_to_world_to_screen(cameraToWorld);
}

TEST(InverseDepthPoint, convertBackAndForthCenter)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const utils::ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);
    EXPECT_NEAR(inverseDepth.get_inverse_depth(), 0.0, 0.001);
    EXPECT_NEAR(inverseDepth.get_theta(), M_PI / 2.0, 0.0001);
    EXPECT_NEAR(inverseDepth.get_phi(), 0.0, 0.0001);

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), observation.x(), 0.01);
    EXPECT_NEAR(screenCoordinates.y(), observation.y(), 0.01);
}

TEST(InverseDepthPoint, convertBackAndForthTopLeft)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const utils::ScreenCoordinate2D observation(vector2::Zero());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);
    EXPECT_NEAR(inverseDepth.get_inverse_depth(), 0.0, 0.001);
    EXPECT_NEAR(inverseDepth.get_theta(), 1.2101243885134101, 0.0001); // left
    EXPECT_NEAR(inverseDepth.get_phi(), 0.52694322718942976, 0.0001);  // top

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), observation.x(), 0.01);
    EXPECT_NEAR(screenCoordinates.y(), observation.y(), 0.01);
}

TEST(InverseDepthPoint, convertBackAndForthBottomRight)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const auto& imageSize = Parameters::get_camera_1_image_size();

    // observe the center of the camera
    const utils::ScreenCoordinate2D observation(imageSize.x(), imageSize.y());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);
    EXPECT_NEAR(inverseDepth.get_inverse_depth(), 0.0, 0.001);
    EXPECT_NEAR(inverseDepth.get_theta(), 1.931468265076383, 0.0001);  // right
    EXPECT_NEAR(inverseDepth.get_phi(), -0.52694322718942932, 0.0001); // bottom

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), observation.x(), 0.01);
    EXPECT_NEAR(screenCoordinates.y(), observation.y(), 0.01);
}

TEST(InverseDepthPoint, convertBackAndForthCenterWithTransfoX)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const utils::ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w =
            utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(250, 0.0, 0.0));

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);
    EXPECT_NEAR(inverseDepth.get_inverse_depth(), 0.0, 0.001);
    EXPECT_NEAR(inverseDepth.get_theta(), M_PI / 2.0, 0.0001);
    EXPECT_NEAR(inverseDepth.get_phi(), 0.0, 0.0001);

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), observation.x(), 0.01);
    EXPECT_NEAR(screenCoordinates.y(), observation.y(), 0.01);
}

TEST(InverseDepthPoint, convertBackAndForthCenterWithTransfoY)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const utils::ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w =
            utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(0.0, 250, 0.0));

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);
    EXPECT_NEAR(inverseDepth.get_inverse_depth(), 0.0, 0.001);
    EXPECT_NEAR(inverseDepth.get_theta(), M_PI / 2.0, 0.0001);
    EXPECT_NEAR(inverseDepth.get_phi(), 0.0, 0.0001);

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), observation.x(), 0.01);
    EXPECT_NEAR(screenCoordinates.y(), observation.y(), 0.01);
}

TEST(InverseDepthPoint, convertBackAndForthCenterWithTransfoZ)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const utils::ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w =
            utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(0.0, 0.0, 250));

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);
    EXPECT_NEAR(inverseDepth.get_inverse_depth(), 0.0, 0.001);
    EXPECT_NEAR(inverseDepth.get_theta(), M_PI / 2.0, 0.0001);
    EXPECT_NEAR(inverseDepth.get_phi(), 0.0, 0.0001);

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), observation.x(), 0.01);
    EXPECT_NEAR(screenCoordinates.y(), observation.y(), 0.01);
}

TEST(InverseDepthPoint, convertBackAndForthCenterWithTransfo)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const utils::ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w =
            utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(250, 150, 300));

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);
    EXPECT_NEAR(inverseDepth.get_inverse_depth(), 0.0, 0.001);
    EXPECT_NEAR(inverseDepth.get_theta(), M_PI / 2.0, 0.0001);
    EXPECT_NEAR(inverseDepth.get_phi(), 0.0, 0.0001);

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), observation.x(), 0.01);
    EXPECT_NEAR(screenCoordinates.y(), observation.y(), 0.01);
}

TEST(InverseDepthPoint, convertBackAndForthTopLeftWithTransfo)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const utils::ScreenCoordinate2D observation(vector2::Zero());
    const CameraToWorldMatrix& c2w =
            utils::compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(250, 150, 300));

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);
    EXPECT_NEAR(inverseDepth.get_inverse_depth(), 0.0, 0.001);
    EXPECT_NEAR(inverseDepth.get_theta(), 0.62879628641543905, 0.0001); // left
    EXPECT_NEAR(inverseDepth.get_phi(), -2.4980915447965168, 0.0001);   // top

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), observation.x(), 0.01);
    EXPECT_NEAR(screenCoordinates.y(), observation.y(), 0.01);
}

TEST(InverseDepthPoint, convertBackAndForthBottomRightWithTransfo)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const auto& imageSize = Parameters::get_camera_1_image_size();

    // observe the center of the camera
    const utils::ScreenCoordinate2D observation(imageSize.x(), imageSize.y());
    const CameraToWorldMatrix& c2w =
            utils::compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(250, 150, 300));

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);
    EXPECT_NEAR(inverseDepth.get_inverse_depth(), 0.0, 0.001);
    EXPECT_NEAR(inverseDepth.get_theta(), 0.62879628641543905, 0.0001); // right
    EXPECT_NEAR(inverseDepth.get_phi(), 0.6435011087932766, 0.0001);    // bottom

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), observation.x(), 0.01);
    EXPECT_NEAR(screenCoordinates.y(), observation.y(), 0.01);
}

TEST(InverseDepthPoint, convertBackAndForthCenterWithRotation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const utils::ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform_no_correction(
            quaternion(0.246242, -0.312924, -0.896867, 0.189256), vector3::Zero());

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);
    EXPECT_NEAR(inverseDepth.get_inverse_depth(), 0.0, 0.001);
    EXPECT_NEAR(inverseDepth.get_theta(), 2.5088563012979175, 0.0001);
    EXPECT_NEAR(inverseDepth.get_phi(), -2.8220093192865892, 0.001);

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), observation.x(), 0.01);
    EXPECT_NEAR(screenCoordinates.y(), observation.y(), 0.01);
}

TEST(InverseDepthPoint, convertBackAndForthTopLeftWithRotation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const utils::ScreenCoordinate2D observation(vector2::Zero());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform_no_correction(
            quaternion(0.246242, -0.312924, -0.896867, 0.189256), vector3::Zero());

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);
    EXPECT_NEAR(inverseDepth.get_inverse_depth(), 0.0, 0.001);
    EXPECT_NEAR(inverseDepth.get_theta(), 2.2516929542517667, 0.0001); // left
    EXPECT_NEAR(inverseDepth.get_phi(), -1.965061641221322, 0.0001);   // top

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), observation.x(), 0.01);
    EXPECT_NEAR(screenCoordinates.y(), observation.y(), 0.01);
}

TEST(InverseDepthPoint, convertBackAndForthBottomRightWithRotation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const auto& imageSize = Parameters::get_camera_1_image_size();

    // observe the center of the camera
    const utils::ScreenCoordinate2D observation(imageSize.x(), imageSize.y());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform_no_correction(
            quaternion(0.246242, -0.312924, -0.896867, 0.189256), vector3::Zero());

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);
    EXPECT_NEAR(inverseDepth.get_inverse_depth(), 0.0, 0.001);
    EXPECT_NEAR(inverseDepth.get_theta(), 2.3106902074817603, 0.0001); // right
    EXPECT_NEAR(inverseDepth.get_phi(), 2.5408706082452479, 0.0001);   // bottom

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), imageSize.x(), 0.01);
    EXPECT_NEAR(screenCoordinates.y(), imageSize.y(), 0.01);
}

TEST(InverseDepthPointFusion, centerPointParallelFusion)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const utils::ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());

    // convert to inverse
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix33::Identity(), cv::Mat());

    const auto beforeMergeInverseDepthCoord = inverseDepth._coordinates;

    EXPECT_NEAR(beforeMergeInverseDepthCoord.get_inverse_depth(), 0.0, 0.001);
    EXPECT_NEAR(beforeMergeInverseDepthCoord.get_theta(), 0.0, M_PI / 2.0);
    EXPECT_NEAR(beforeMergeInverseDepthCoord.get_phi(), 0.0, 0.001);

    Eigen::Matrix<double, 3, 6> toCartesianJacobian;
    const WorldCoordinate& firstCartesian = beforeMergeInverseDepthCoord.to_world_coordinates(toCartesianJacobian);
    EXPECT_NEAR(firstCartesian.x(), 2000, 0.1);
    EXPECT_NEAR(firstCartesian.y(), 0.0, 0.1);
    EXPECT_NEAR(firstCartesian.z(), 0.0, 0.1);

    // store the covariance at the start of the process
    const tracking::PointInverseDepth::Covariance& beforeMergeInverseCov = inverseDepth._covariance;
    const WorldCoordinateCovariance& beforeMergeCovariance =
            tracking::PointInverseDepth::compute_cartesian_covariance(beforeMergeInverseCov, toCartesianJacobian);
    EXPECT_TRUE(is_covariance_valid(beforeMergeInverseCov));
    EXPECT_TRUE(is_covariance_valid(beforeMergeCovariance));

    EXPECT_GT(beforeMergeCovariance(0, 0),
              1e5); // high variance for x coordinate (forward) very high variance for z coordinate depth is unknown
    EXPECT_GT(beforeMergeCovariance(1, 1), 100); // high variance for y coordinate (left) : angle theta as some variance
    EXPECT_GT(beforeMergeCovariance(2, 2), 100); // high variance for y coordinate (left) : angle phi as some variance

    /**
     ** add a new measurment at 90 degrees on the side
     */

    const CameraToWorldMatrix& c2wSide90 = utils::compute_camera_to_world_transform(
            get_quaternion_from_euler_angles(EulerAngles(0.0, -90 * EulerToRadian, 0.0)), vector3(1000, 0.0, 1000.0));
    // fuse the two points
    EXPECT_TRUE(inverseDepth.track(observation, c2wSide90, matrix33::Identity(), cv::Mat()));

    const auto finalPose = inverseDepth._coordinates.to_world_coordinates();
    const auto finalPoseCovariance = tracking::PointInverseDepth::compute_cartesian_covariance(
            inverseDepth._coordinates, inverseDepth._covariance);
    EXPECT_TRUE(is_covariance_valid(finalPoseCovariance));

    // final pose is close to the expected (1000, 0, 0)
    EXPECT_NEAR(finalPose.x(), 1000.0, 10.0);
    EXPECT_NEAR(finalPose.y(), 0.0, 1.0);
    EXPECT_NEAR(finalPose.z(), 0.0, 1.0);
}

void estimate_plane_error(const PlaneCoordinates& planeA, const PlaneCoordinates& planeB)
{
    const vector3& normalA = planeA.get_normal();
    const vector3& normalB = planeB.get_normal();

    EXPECT_NEAR(normalA.x(), normalB.x(), 0.001);
    EXPECT_NEAR(normalA.y(), normalB.y(), 0.001);
    EXPECT_NEAR(normalA.z(), normalB.z(), 0.001);
    EXPECT_NEAR(planeA.get_d(), planeB.get_d(), 15); // renormalization error accumulation
}

void test_plane_set_camera_to_world_to_camera(const CameraToWorldMatrix& cameraToWorld)
{
    const PlaneCameraToWorldMatrix& planeCameraToWorld = compute_plane_camera_to_world_matrix(cameraToWorld);
    const PlaneWorldToCameraMatrix& planeWorldToCamera =
            compute_plane_world_to_camera_matrix(compute_world_to_camera_transform(cameraToWorld));
    const double normalXIter = 0.3;
    const double normalYIter = 0.1;
    const double normalZIter = 0.1;

    for (double x = 1; x <= 1.0; x += normalXIter)
    {
        for (double y = -1; y < 1.0; y += normalYIter)
        {
            for (double z = -1; z < 1.0; z += normalZIter)
            {
                const vector3 planeNormal = vector3(x, y, z).normalized();
                for (double d = 1; d < 100; d += 5.5)
                {
                    const PlaneCameraCoordinates originalCameraPlane(planeNormal, d);
                    const PlaneWorldCoordinates worldPlane =
                            originalCameraPlane.to_world_coordinates(planeCameraToWorld);
                    const PlaneCameraCoordinates newCameraCoordinates =
                            worldPlane.to_camera_coordinates(planeWorldToCamera);

                    estimate_plane_error(originalCameraPlane, newCameraCoordinates);
                }
            }
        }
    }
}

TEST(PlaneCoordinateSystemTests, ScreenToWorldToScreenAtOrigin)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(0, 0, 0));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraFarFromOrigin)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(-100, 1000, 100));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraRotation1)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform_no_correction(quaternion(0.3, 0.2, 0.1, 0.4), vector3(0, 0, 0));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraRotation2)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform_no_correction(quaternion(0.6, 0.1, 0.2, 0.1), vector3(0, 0, 0));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraRotation3)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform_no_correction(quaternion(0.6, 0.1, 0.2, 0.1), vector3(100, -100, -100));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

/**
 *      Test the line distance function
 */

TEST(LineDistances, LineDistancesAtZero)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // origin forward
    const vector3 point1(0.0, 0.0, 0.0);
    const vector3 normal1(1.0, 0.0, 0.0);

    // test forward and backward normal
    ASSERT_NEAR(signed_line_distance<3>(point1, normal1, point1, normal1).norm(), 0.0, 0.0001);
    ASSERT_NEAR(signed_line_distance<3>(point1, normal1, point1, -normal1).norm(), 0.0, 0.0001);
    ASSERT_NEAR(signed_line_distance<3>(point1, -normal1, point1, -normal1).norm(), 0.0, 0.0001);
    ASSERT_NEAR(signed_line_distance<3>(point1, -normal1, point1, normal1).norm(), 0.0, 0.0001);

    // point further on x
    const vector3 point2(1000.0, 0.0, 0.0);
    const vector3 normal2(1.0, 0.0, 0.0);
    // test forward and backward normal, no difference
    ASSERT_NEAR(signed_line_distance<3>(point1, normal1, point2, normal2).norm(), 0.0, 0.0001);
    ASSERT_NEAR(signed_line_distance<3>(point1, normal1, point2, -normal2).norm(), 0.0, 0.0001);
    ASSERT_NEAR(signed_line_distance<3>(point1, -normal1, point2, -normal2).norm(), 0.0, 0.0001);
    ASSERT_NEAR(signed_line_distance<3>(point1, -normal1, point2, normal2).norm(), 0.0, 0.0001);

    // point looking sideway
    const vector3 point3(1000.0, 0.0, 0.0);
    const vector3 normal3(0.0, 1.0, 0.0);
    // test forward and backward normal, no difference
    ASSERT_NEAR(signed_line_distance<3>(point1, normal1, point3, normal3).norm(), 0.0, 0.0001);
    ASSERT_NEAR(signed_line_distance<3>(point1, normal1, point3, -normal3).norm(), 0.0, 0.0001);
    ASSERT_NEAR(signed_line_distance<3>(point1, -normal1, point3, -normal3).norm(), 0.0, 0.0001);
    ASSERT_NEAR(signed_line_distance<3>(point1, -normal1, point3, normal3).norm(), 0.0, 0.0001);

    // point looking down
    const vector3 point4(1000.0, 0.0, 0.0);
    const vector3 normal4(0.0, 0.0, 1.0);
    // test forward and backward normal, no difference
    ASSERT_NEAR(signed_line_distance<3>(point1, normal1, point4, normal4).norm(), 0.0, 0.0001);
    ASSERT_NEAR(signed_line_distance<3>(point1, normal1, point4, -normal4).norm(), 0.0, 0.0001);
    ASSERT_NEAR(signed_line_distance<3>(point1, -normal1, point4, -normal4).norm(), 0.0, 0.0001);
    ASSERT_NEAR(signed_line_distance<3>(point1, -normal1, point4, normal4).norm(), 0.0, 0.0001);
}

TEST(LineDistances, LineDistances)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // origin forward
    const vector3 point1(0.0, 0.0, 0.0);
    const vector3 normal1(1.0, 0.0, 0.0);

    // high point, looking left
    const vector3 point2(0.0, 0.0, 1000.0);
    const vector3 normal2(0.0, 1.0, 0.0);

    // test forward and backward normal
    ASSERT_NEAR(signed_line_distance<3>(point1, normal1, point2, normal2).norm(), 1000.0, 0.0001);
    ASSERT_NEAR(signed_line_distance<3>(point1, normal1, point2, -normal2).norm(), 1000.0, 0.0001);
    ASSERT_NEAR(signed_line_distance<3>(point1, -normal1, point2, -normal2).norm(), 1000.0, 0.0001);
    ASSERT_NEAR(signed_line_distance<3>(point1, -normal1, point2, normal2).norm(), 1000.0, 0.0001);

    // shifted parralel point
    const vector3 point3(0.0, 0.0, 1000.0);
    const vector3 normal3(1.0, 0.0, 0.0);

    // test forward and backward normal
    ASSERT_NEAR(signed_line_distance<3>(point1, normal1, point3, normal3).norm(), 1000.0, 0.0001);
    ASSERT_NEAR(signed_line_distance<3>(point1, normal1, point3, -normal3).norm(), 1000.0, 0.0001);
    ASSERT_NEAR(signed_line_distance<3>(point1, -normal1, point3, -normal3).norm(), 1000.0, 0.0001);
    ASSERT_NEAR(signed_line_distance<3>(point1, -normal1, point3, normal3).norm(), 1000.0, 0.0001);
}

} // namespace rgbd_slam::utils