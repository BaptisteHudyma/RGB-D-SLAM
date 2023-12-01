#include "angle_utils.hpp"
#include "covariances.hpp"
#include "parameters.hpp"
#include "point_with_tracking.hpp"
#include "types.hpp"
#include "utils/camera_transformation.hpp"
#include "utils/coordinates/point_coordinates.hpp"
#include "utils/coordinates/plane_coordinates.hpp"
#include <gtest/gtest.h>

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
            compute_camera_to_world_transform(quaternion::Identity(), vector3(0, 0, 0));

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
            compute_camera_to_world_transform(quaternion::Identity(), vector3(-100, 100, 200));

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
            compute_camera_to_world_transform(quaternion(0.0, 1.0, 0.0, 0.0), vector3(0, 0, 0));

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
            compute_camera_to_world_transform(quaternion(0.0, 0.0, 1.0, 0.0), vector3(0, 0, 0));

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
            compute_camera_to_world_transform(quaternion(0.0, 0, 0, 1.0), vector3(0, 0, 0));

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
            compute_camera_to_world_transform(quaternion(0.5, 0.5, 0.5, 0.5), vector3(0, 0, 0));

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
            compute_camera_to_world_transform(quaternion(0.0, 1.0, 0.0, 0.0), vector3(-100, 100, 200));

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
            compute_camera_to_world_transform(quaternion(0.0, 1.0, 0.0, 0.0), vector3(0, 0, 0));

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
            compute_camera_to_world_transform(quaternion::Identity(), vector3(0, 0, 0));
    test_point_set_screen_to_world_to_screen(cameraToWorld);
}

TEST(PointCoordinateSystemTests, ScreenToWorldToScreenFarFromOrigin)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform(quaternion::Identity(), vector3(-100, 1000, 100));
    test_point_set_screen_to_world_to_screen(cameraToWorld);
}

TEST(PointCoordinateSystemTests, ScreenToWorldToScreenRotation1)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform(quaternion(0.3, 0.2, 0.1, 0.4), vector3(0, 0, 0));
    test_point_set_screen_to_world_to_screen(cameraToWorld);
}

TEST(PointCoordinateSystemTests, ScreenToWorldToScreenRotation2)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform(quaternion(0.6, 0.1, 0.2, 0.1), vector3(0, 0, 0));
    test_point_set_screen_to_world_to_screen(cameraToWorld);
}

TEST(PointCoordinateSystemTests, ScreenToWorldToScreenRotation3)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform(quaternion(0.6, 0.1, 0.2, 0.1), vector3(100, -100, -100));
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
    EXPECT_NEAR(inverseDepth._inverseDepth_mm, 0.0, 0.001);
    EXPECT_NEAR(inverseDepth._phi_rad, 0.0, 0.0001);
    EXPECT_NEAR(inverseDepth._theta_rad, 0.0, 0.0001);

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), Parameters::get_camera_1_center().x(), 0.01);
    EXPECT_NEAR(screenCoordinates.y(), Parameters::get_camera_1_center().y(), 0.01);
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
    EXPECT_NEAR(inverseDepth._inverseDepth_mm, 0.0, 0.001);
    EXPECT_NEAR(inverseDepth._phi_rad, 0.36067193828148641, 0.0001);    // top
    EXPECT_NEAR(inverseDepth._theta_rad, -0.52694322718942965, 0.0001); // left

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), 0.0, 0.01);
    EXPECT_NEAR(screenCoordinates.y(), 0.0, 0.01);
}

TEST(InverseDepthPoint, convertBackAndForthBottomRight)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const auto& imageSize = Parameters::get_camera_1_image_size();
    const double imageHeight = imageSize.x();
    const double imageWidth = imageSize.y();

    // observe the center of the camera
    const utils::ScreenCoordinate2D observation(imageHeight, imageWidth);
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);
    EXPECT_NEAR(inverseDepth._inverseDepth_mm, 0.0, 0.001);
    EXPECT_NEAR(inverseDepth._phi_rad, -0.36067193828148641, 0.0001);  // bottom
    EXPECT_NEAR(inverseDepth._theta_rad, 0.52694322718942965, 0.0001); // right

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), imageHeight, 0.01);
    EXPECT_NEAR(screenCoordinates.y(), imageWidth, 0.01);
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
    EXPECT_NEAR(inverseDepth._inverseDepth_mm, 0.0, 0.001);
    EXPECT_NEAR(inverseDepth._phi_rad, 0.0, 0.0001);
    EXPECT_NEAR(inverseDepth._theta_rad, 0.0, 0.0001);

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), Parameters::get_camera_1_center().x(), 0.01);
    EXPECT_NEAR(screenCoordinates.y(), Parameters::get_camera_1_center().y(), 0.01);
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
            utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(250, 150, 300));

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);
    EXPECT_NEAR(inverseDepth._inverseDepth_mm, 0.0, 0.001);
    EXPECT_NEAR(inverseDepth._phi_rad, 0.36067193828148641, 0.0001);    // top
    EXPECT_NEAR(inverseDepth._theta_rad, -0.52694322718942965, 0.0001); // left

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), 0.0, 0.01);
    EXPECT_NEAR(screenCoordinates.y(), 0.0, 0.01);
}

TEST(InverseDepthPoint, convertBackAndForthBottomRightWithTransfo)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const auto& imageSize = Parameters::get_camera_1_image_size();
    const double imageHeight = imageSize.x();
    const double imageWidth = imageSize.y();

    // observe the center of the camera
    const utils::ScreenCoordinate2D observation(imageHeight, imageWidth);
    const CameraToWorldMatrix& c2w =
            utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(250, 150, 300));

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);
    EXPECT_NEAR(inverseDepth._inverseDepth_mm, 0.0, 0.001);
    EXPECT_NEAR(inverseDepth._phi_rad, -0.36067193828148641, 0.0001);  // bottom
    EXPECT_NEAR(inverseDepth._theta_rad, 0.52694322718942965, 0.0001); // right

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), imageHeight, 0.01);
    EXPECT_NEAR(screenCoordinates.y(), imageWidth, 0.01);
}

TEST(InverseDepthPoint, convertBackAndForthCenterWithRotation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const utils::ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(
            quaternion(0.246242, -0.312924, -0.896867, 0.189256), vector3::Zero());

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);
    EXPECT_NEAR(inverseDepth._inverseDepth_mm, 0.0, 0.001);
    EXPECT_NEAR(inverseDepth._phi_rad, 0.18687189393893863, 0.001);
    EXPECT_NEAR(inverseDepth._theta_rad, -2.5334324872460674, 0.0001);

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), Parameters::get_camera_1_center().x(), 0.01);
    EXPECT_NEAR(screenCoordinates.y(), Parameters::get_camera_1_center().y(), 0.01);
}

TEST(InverseDepthPoint, convertBackAndForthTopLeftWithRotation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const utils::ScreenCoordinate2D observation(vector2::Zero());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(
            quaternion(0.246242, -0.312924, -0.896867, 0.189256), vector3::Zero());

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);
    EXPECT_NEAR(inverseDepth._inverseDepth_mm, 0.0, 0.001);
    EXPECT_NEAR(inverseDepth._phi_rad, 0.80005693584845172, 0.0001);   // top
    EXPECT_NEAR(inverseDepth._theta_rad, -2.6988385480625769, 0.0001); // left

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), 0.0, 0.01);
    EXPECT_NEAR(screenCoordinates.y(), 0.0, 0.01);
}

TEST(InverseDepthPoint, convertBackAndForthBottomRightWithRotation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const auto& imageSize = Parameters::get_camera_1_image_size();
    const double imageHeight = imageSize.x();
    const double imageWidth = imageSize.y();

    // observe the center of the camera
    const utils::ScreenCoordinate2D observation(imageHeight, imageWidth);
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(
            quaternion(0.246242, -0.312924, -0.896867, 0.189256), vector3::Zero());

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);
    EXPECT_NEAR(inverseDepth._inverseDepth_mm, 0.0, 0.001);
    EXPECT_NEAR(inverseDepth._phi_rad, -0.43063853453197076, 0.0001);  // bottom
    EXPECT_NEAR(inverseDepth._theta_rad, -2.4067705585346779, 0.0001); // right

    // retroproject to screen
    utils::ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_screen_coordinates(utils::compute_world_to_camera_transform(c2w), screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), imageHeight, 0.01);
    EXPECT_NEAR(screenCoordinates.y(), imageWidth, 0.01);
}

TEST(InverseDepthPointFusion, centerPointParallelFusion)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const utils::ScreenCoordinate2D observation(Parameters::get_camera_1_center() * 1.00001);
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());

    // convert to inverse
    tracking::PointInverseDepth inverseDepth(observation, c2w, matrix33::Zero(), cv::Mat());
    EXPECT_NEAR(inverseDepth._coordinates._inverseDepth_mm, 0.0, 0.001);
    EXPECT_NEAR(inverseDepth._coordinates._phi_rad, 0.0, 0.001);
    EXPECT_NEAR(inverseDepth._coordinates._theta_rad, 0.0, 0.001);

    const auto beforeMergeInverseCov = inverseDepth._covariance;
    const auto& beforeMergeCovariance = tracking::PointInverseDepth::compute_cartesian_covariance(
            inverseDepth._coordinates, inverseDepth._covariance);

    EXPECT_TRUE(is_covariance_valid(beforeMergeCovariance));
    EXPECT_GT(beforeMergeCovariance(0, 0),
              1e3); // very high variance for x coordinate (center of observation) : angle theta as some variance
    EXPECT_GT(beforeMergeCovariance(1, 1),
              1e3); // very high variance for x coordinate (center of observation) : angle theta as some variance
    EXPECT_GT(beforeMergeCovariance(2, 2), 1e3); // very high variance for z coordinate depth is unknown

    // fuse the two points
    EXPECT_TRUE(inverseDepth.track(observation, c2w, matrix33::Zero(), cv::Mat()));

    const auto afterMergeInverseCov = inverseDepth._covariance;
    const auto& afterMergeCovariance = tracking::PointInverseDepth::compute_cartesian_covariance(
            inverseDepth._coordinates, inverseDepth._covariance);

    EXPECT_TRUE(is_covariance_valid(afterMergeCovariance));
    EXPECT_LT(afterMergeInverseCov(3, 3), beforeMergeInverseCov(3, 3));
    EXPECT_LT(afterMergeInverseCov(4, 4), beforeMergeInverseCov(4, 4));
    EXPECT_LT(afterMergeInverseCov(5, 5), beforeMergeInverseCov(5, 5));

    // add a new measurment at 90 degrees
    const CameraToWorldMatrix& c2w90 = utils::compute_camera_to_world_transform(
            get_quaternion_from_euler_angles(EulerAngles(0.0, -90 * EulerToRadian, 0.0)), vector3(1000, 0, 1000.0));
    // fuse the two points
    EXPECT_TRUE(inverseDepth.track(observation, c2w90, matrix33::Zero(), cv::Mat()));

    const CameraToWorldMatrix& c2wMinus90 = utils::compute_camera_to_world_transform(
            get_quaternion_from_euler_angles(EulerAngles(0.0, 90 * EulerToRadian, 0.0)), vector3(-10000, 0, 1000.0));
    // fuse the two points
    EXPECT_TRUE(inverseDepth.track(observation, c2wMinus90, matrix33::Zero(), cv::Mat()));

    const auto& finalCovariance = tracking::PointInverseDepth::compute_cartesian_covariance(inverseDepth._coordinates,
                                                                                            inverseDepth._covariance);

    std::cout << "---------------------------" << std::endl << std::endl;
    std::cout << finalCovariance << std::endl << std::endl;

    // result should be around (0, 0, 1000)
    std::cout << inverseDepth._coordinates.to_world_coordinates().transpose() << std::endl;

    EXPECT_TRUE(is_covariance_valid(finalCovariance));
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
            compute_camera_to_world_transform(quaternion::Identity(), vector3(0, 0, 0));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraFarFromOrigin)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform(quaternion::Identity(), vector3(-100, 1000, 100));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraRotation1)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform(quaternion(0.3, 0.2, 0.1, 0.4), vector3(0, 0, 0));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraRotation2)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform(quaternion(0.6, 0.1, 0.2, 0.1), vector3(0, 0, 0));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraRotation3)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            compute_camera_to_world_transform(quaternion(0.6, 0.1, 0.2, 0.1), vector3(100, -100, -100));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

} // namespace rgbd_slam::utils