#include "angle_utils.hpp"
#include "covariances.hpp"
#include "distance_utils.hpp"
#include "line.hpp"
#include "parameters.hpp"
#include "inverse_depth_with_tracking.hpp"
#include "types.hpp"
#include "utils/camera_transformation.hpp"
#include "coordinates/inverse_depth_coordinates.hpp"
#include "coordinates/point_coordinates.hpp"
#include "coordinates/plane_coordinates.hpp"
#include <gtest/gtest.h>
#include <iostream>

namespace rgbd_slam {

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
            utils::compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(0, 0, 0));

    const matrix44& tr = get_transformation_matrix(
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
            utils::compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(-100, 100, 200));

    const matrix44& tr = get_transformation_matrix(vector3(1, 0, 0),
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
            utils::compute_camera_to_world_transform_no_correction(quaternion(0.0, 1.0, 0.0, 0.0), vector3(0, 0, 0));

    const matrix44& tr = get_transformation_matrix(
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
            utils::compute_camera_to_world_transform_no_correction(quaternion(0.0, 0.0, 1.0, 0.0), vector3(0, 0, 0));

    const matrix44& tr = get_transformation_matrix(
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
            utils::compute_camera_to_world_transform_no_correction(quaternion(0.0, 0, 0, 1.0), vector3(0, 0, 0));

    const matrix44& tr = get_transformation_matrix(
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
            utils::compute_camera_to_world_transform_no_correction(quaternion(0.5, 0.5, 0.5, 0.5), vector3(0, 0, 0));

    const matrix44& tr = get_transformation_matrix(
            vector3(1, 0, 0), vector3(0, 1, 0), vector3::Zero(), vector3(0, 1, 0), vector3(0, 0, 1), vector3::Zero());

    EXPECT_TRUE(cameraToWorld.isApprox(tr));
}

TEST(CoordinateSystemChangeTests, CameraToWorldFarFromOriginWithRotation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld = utils::compute_camera_to_world_transform_no_correction(
            quaternion(0.0, 1.0, 0.0, 0.0), vector3(-100, 100, 200));

    const matrix44& tr = get_transformation_matrix(vector3(1, 0, 0),
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
            utils::compute_camera_to_world_transform_no_correction(quaternion(0.0, 1.0, 0.0, 0.0), vector3(0, 0, 0));

    const matrix44& tr = get_transformation_matrix(vector3(1, 0, 0),
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
    const WorldToCameraMatrix worldToCamera = utils::compute_world_to_camera_transform(cameraToWorld);

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
            utils::compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(0, 0, 0));
    test_point_set_screen_to_world_to_screen(cameraToWorld);
}

TEST(PointCoordinateSystemTests, ScreenToWorldToScreenFarFromOrigin)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(-100, 1000, 100));
    test_point_set_screen_to_world_to_screen(cameraToWorld);
}

TEST(PointCoordinateSystemTests, ScreenToWorldToScreenRotation1)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion(0.3, 0.2, 0.1, 0.4), vector3(0, 0, 0));
    test_point_set_screen_to_world_to_screen(cameraToWorld);
}

TEST(PointCoordinateSystemTests, ScreenToWorldToScreenRotation2)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion(0.6, 0.1, 0.2, 0.1), vector3(0, 0, 0));
    test_point_set_screen_to_world_to_screen(cameraToWorld);
}

TEST(PointCoordinateSystemTests, ScreenToWorldToScreenRotation3)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld = utils::compute_camera_to_world_transform_no_correction(
            quaternion(0.6, 0.1, 0.2, 0.1), vector3(100, -100, -100));
    test_point_set_screen_to_world_to_screen(cameraToWorld);
}

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

TEST(InverseDepthPoint, convertBackAndForthCenter)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthTopLeft)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(vector2::Zero());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthBottomRight)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const auto& imageSize = Parameters::get_camera_1_image_size();

    // observe the center of the camera
    const ScreenCoordinate2D observation(imageSize.x(), imageSize.y());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthCenterWithTransfoX)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w =
            utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(2500, 0.0, 0.0));

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthCenterWithTransfoY)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w =
            utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(0.0, 2500, 0.0));

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthCenterWithTransfoZ)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w =
            utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(0.0, 0.0, 2500));

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthCenterWithTransfo)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w =
            utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(2500, 1500, 3000));

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthTopLeftWithTransfo)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(vector2::Zero());
    const CameraToWorldMatrix& c2w =
            utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(2500, 1500, 3000));

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthBottomRightWithTransfo)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const auto& imageSize = Parameters::get_camera_1_image_size();

    // observe the center of the camera
    const ScreenCoordinate2D observation(imageSize.x(), imageSize.y());
    const CameraToWorldMatrix& c2w =
            utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(2500, 1500, 3000));

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthCenterWithRotation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(
            quaternion(0.246242, -0.312924, -0.896867, 0.189256), vector3::Zero());

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthTopLeftWithRotation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(vector2::Zero());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(
            quaternion(0.246242, -0.312924, -0.896867, 0.189256), vector3::Zero());

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthBottomRightWithRotation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const auto& imageSize = Parameters::get_camera_1_image_size();

    // observe the center of the camera
    const ScreenCoordinate2D observation(imageSize.x(), imageSize.y());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(
            quaternion(0.246242, -0.312924, -0.896867, 0.189256), vector3::Zero());

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthTopLeftWithRotationTranslation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(0.0, 0.0);
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(
            quaternion(0.246242, -0.312924, -0.896867, 0.189256), vector3(2500, 1500, 3000));

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthBottomRightWithRotationTranslation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const auto& imageSize = Parameters::get_camera_1_image_size();

    // observe the center of the camera
    const ScreenCoordinate2D observation(imageSize.x(), imageSize.y());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(
            quaternion(0.246242, -0.312924, -0.896867, 0.189256), vector3(2500, 1500, 3000));

    assert_inverse_point_back_proj(c2w, observation);
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
    const PlaneCameraToWorldMatrix& planeCameraToWorld = utils::compute_plane_camera_to_world_matrix(cameraToWorld);
    const PlaneWorldToCameraMatrix& planeWorldToCamera =
            utils::compute_plane_world_to_camera_matrix(utils::compute_world_to_camera_transform(cameraToWorld));
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
            utils::compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(0, 0, 0));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraFarFromOrigin)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(-100, 1000, 100));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraRotation1)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion(0.3, 0.2, 0.1, 0.4), vector3(0, 0, 0));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraRotation2)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion(0.6, 0.1, 0.2, 0.1), vector3(0, 0, 0));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraRotation3)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld = utils::compute_camera_to_world_transform_no_correction(
            quaternion(0.6, 0.1, 0.2, 0.1), vector3(100, -100, -100));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

/**
 *      Test the point to line distance function
 */

TEST(PointToLine2dDistances, LineDistancesAtZero)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // origin forward
    const vector2 point1(0.0, 0.0);
    const vector2 normal1(1.0, 0.0);

    utils::Line<2> line1(point1, normal1);

    ASSERT_NEAR(line1.distance(vector2(0.0, 0.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(0.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(1000.0, 0.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(1000.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(-1000.0, 0.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(-1000.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(0.0, 1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(0.0, 1000.0)).y(), 1000.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(0.0, -1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(0.0, -1000.0)).y(), -1000.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(1000.0, -1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(1000.0, -1000.0)).y(), -1000.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(1000.0, 1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(1000.0, 1000.0)).y(), 1000.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(-1000.0, -1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(-1000.0, -1000.0)).y(), -1000.0, 0.0001);

    // origin backward
    const vector2 point2(0.0, 0.0);
    const vector2 normal2(-1.0, 0.0);

    utils::Line<2> line2(point2, normal2);

    ASSERT_NEAR(line2.distance(vector2(0.0, 0.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line2.distance(vector2(0.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line2.distance(vector2(1000.0, 0.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line2.distance(vector2(1000.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line2.distance(vector2(-1000.0, 0.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line2.distance(vector2(-1000.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line2.distance(vector2(0.0, 1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line2.distance(vector2(0.0, 1000.0)).y(), 1000.0, 0.0001);

    ASSERT_NEAR(line2.distance(vector2(0.0, -1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line2.distance(vector2(0.0, -1000.0)).y(), -1000.0, 0.0001);

    ASSERT_NEAR(line2.distance(vector2(1000.0, -1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line2.distance(vector2(1000.0, -1000.0)).y(), -1000.0, 0.0001);

    ASSERT_NEAR(line2.distance(vector2(1000.0, 1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line2.distance(vector2(1000.0, 1000.0)).y(), 1000.0, 0.0001);

    ASSERT_NEAR(line2.distance(vector2(-1000.0, -1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line2.distance(vector2(-1000.0, -1000.0)).y(), -1000.0, 0.0001);

    // origin right
    const vector2 point3(0.0, 0.0);
    const vector2 normal3(0.0, 1.0);

    utils::Line<2> line3(point3, normal3);

    ASSERT_NEAR(line3.distance(vector2(0.0, 0.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line3.distance(vector2(0.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line3.distance(vector2(1000.0, 0.0)).x(), 1000.0, 0.0001);
    ASSERT_NEAR(line3.distance(vector2(1000.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line3.distance(vector2(-1000.0, 0.0)).x(), -1000.0, 0.0001);
    ASSERT_NEAR(line3.distance(vector2(-1000.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line3.distance(vector2(0.0, 1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line3.distance(vector2(0.0, 1000.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line3.distance(vector2(0.0, -1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line3.distance(vector2(0.0, -1000.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line3.distance(vector2(1000.0, -1000.0)).x(), 1000.0, 0.0001);
    ASSERT_NEAR(line3.distance(vector2(1000.0, -1000.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line3.distance(vector2(1000.0, 1000.0)).x(), 1000.0, 0.0001);
    ASSERT_NEAR(line3.distance(vector2(1000.0, 1000.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line3.distance(vector2(-1000.0, -1000.0)).x(), -1000.0, 0.0001);
    ASSERT_NEAR(line3.distance(vector2(-1000.0, -1000.0)).y(), 0.0, 0.0001);
}

TEST(PointToLine2dDistances, LineDistancesRotated)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // origin diagonal
    const vector2 point1(100.0, 100.0);
    const vector2 normal1(1.0, 1.0); // pointing right at 45 degrees

    utils::Line<2> line1(point1, normal1);

    ASSERT_NEAR(line1.distance(vector2(0.0, 0.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(0.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(1000.0, 0.0)).x(), 500.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(1000.0, 0.0)).y(), -500.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(-1000.0, 0.0)).x(), -500.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(-1000.0, 0.0)).y(), 500.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(0.0, 1000.0)).x(), -500.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(0.0, 1000.0)).y(), 500.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(0.0, -1000.0)).x(), 500.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(0.0, -1000.0)).y(), -500.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(1000.0, -1000.0)).x(), 1000.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(1000.0, -1000.0)).y(), -1000.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(1000.0, 1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(1000.0, 1000.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(-1000.0, -1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(-1000.0, -1000.0)).y(), 0.0, 0.0001);
}

TEST(PointToLine3Distances, LineDistancesRotated)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // origin diagonal
    const vector3 point1(0.0, 0.0, 0.0);
    const vector3 normal1(1.0, 1.0, 1.0); // pointing right at 45 degrees

    utils::Line<3> line1(point1, normal1);

    ASSERT_NEAR(line1.distance(vector3(0.0, 0.0, 0.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 0.0, 0.0)).y(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 0.0, 0.0)).z(), 0.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(1000.0, 0.0, 0.0)).x(), 666.66666, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(1000.0, 0.0, 0.0)).y(), -333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(1000.0, 0.0, 0.0)).z(), -333.33333, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(-1000.0, 0.0, 0.0)).x(), -666.66666, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(-1000.0, 0.0, 0.0)).y(), 333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(-1000.0, 0.0, 0.0)).z(), 333.33333, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(0.0, 1000.0, 0.0)).x(), -333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 1000.0, 0.0)).y(), 666.66666, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 1000.0, 0.0)).z(), -333.33333, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(0.0, -1000.0, 0.0)).x(), 333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, -1000.0, 0.0)).y(), -666.66666, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, -1000.0, 0.0)).z(), 333.33333, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(0.0, 0.0, 1000.0)).x(), -333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 0.0, 1000.0)).y(), -333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 0.0, 1000.0)).z(), 666.66666, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(0.0, 0.0, -1000.0)).x(), 333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 0.0, -1000.0)).y(), 333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 0.0, -1000.0)).z(), -666.66666, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(0.0, 1000.0, 0.0)).x(), -333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 1000.0, 0.0)).y(), 666.66666, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 1000.0, 0.0)).z(), -333.33333, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(0.0, -1000.0, 0.0)).x(), 333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, -1000.0, 0.0)).y(), -666.66666, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, -1000.0, 0.0)).z(), 333.33333, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(1000.0, -1000.0, 0.0)).x(), 1000.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(1000.0, -1000.0, 0.0)).y(), -1000.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(1000.0, -1000.0, 0.0)).z(), 0.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(1000.0, 1000.0, 1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(1000.0, 1000.0, 1000.0)).y(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(1000.0, 1000.0, 1000.0)).z(), 0.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(-1000.0, -1000.0, -1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(-1000.0, -1000.0, -1000.0)).y(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(-1000.0, -1000.0, -1000.0)).z(), 0.0, 0.0001);
}

/**
 *      Test the line to line distance function
 */

TEST(LineToLineDistances, LineDistancesAtZero)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // origin forward
    const vector3 point1(0.0, 0.0, 0.0);
    const vector3 normal1(1.0, 0.0, 0.0);

    // test forward and backward normal
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point1, normal1).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point1, -normal1).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point1, -normal1).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point1, normal1).norm(), 0.0, 0.0001);

    // point further on x
    const vector3 point2(1000.0, 0.0, 0.0);
    const vector3 normal2(1.0, 0.0, 0.0);
    // test forward and backward normal, no difference
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point2, normal2).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point2, -normal2).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point2, -normal2).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point2, normal2).norm(), 0.0, 0.0001);

    // point looking sideway
    const vector3 point3(1000.0, 0.0, 0.0);
    const vector3 normal3(0.0, 1.0, 0.0);
    // test forward and backward normal, no difference
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point3, normal3).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point3, -normal3).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point3, -normal3).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point3, normal3).norm(), 0.0, 0.0001);

    // point looking down
    const vector3 point4(1000.0, 0.0, 0.0);
    const vector3 normal4(0.0, 0.0, 1.0);
    // test forward and backward normal, no difference
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point4, normal4).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point4, -normal4).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point4, -normal4).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point4, normal4).norm(), 0.0, 0.0001);
}

TEST(LineToLineDistances, LineDistances)
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
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point2, normal2).norm(), 1000.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point2, -normal2).norm(), 1000.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point2, -normal2).norm(), 1000.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point2, normal2).norm(), 1000.0, 0.0001);

    // shifted parralel point
    const vector3 point3(0.0, 0.0, 1000.0);
    const vector3 normal3(1.0, 0.0, 0.0);

    // test forward and backward normal
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point3, normal3).norm(), 1000.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point3, -normal3).norm(), 1000.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point3, -normal3).norm(), 1000.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point3, normal3).norm(), 1000.0, 0.0001);
}

} // namespace rgbd_slam